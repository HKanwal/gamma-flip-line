import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc


def risk_free_rate():
    """Gets the latest 13-month T-bill yield as a decimal value."""
    url = "https://ycharts.com/charts/fund_data.json?calcs=&chartId=&chartType=interactive&correlations=&customGrowthAmount=&dataInLegend=value&dateSelection=range&displayDateRange=false&endDate=&format=real&legendOnChart=false&lineAnnotations=&nameInLegend=name_and_ticker&note=&partner=basic_2000&performanceDisclosure=false&quoteLegend=false&recessions=false&scaleType=linear&securities=id%3AI%3A3MTBRNK%2Cinclude%3Atrue%2C%2C&securityGroup=&securitylistName=&securitylistSecurityId=&source=false&splitType=single&startDate=&title=&units=false&useCustomColors=false&useEstimates=false&zoom=1&hideValueFlags=false&redesign=true&chartAnnotations=&axisExtremes=&sortColumn=&sortDirection=&le_o=quote_page_fund_data&maxPoints=675&chartCreator=false"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    }

    print("Fetching risk-free rate...")
    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        data = res.json()
        risk_free_rate = data["chart_data"][0][0]["raw_data"][-1][1]
        print(f"Risk-free rate successfully fetched: {risk_free_rate}%")
        return risk_free_rate / 100
    else:
        raise Exception(f"Risk-free rate fetch failed with status code {res.status_code}.")


# TODO: Implement 3x retries per request and handle request failure
def highly_shorted_stocks():
    """Returns screened stocks with short interest greater than 20% of float."""

    def extract_stocks(soup: BeautifulSoup):
        return [link.text for link in soup.select("table.screener_table tr td:nth-child(2) a")]

    url = "https://finviz.com/screener.ashx?v=111&f=cap_midover,geo_usa,sh_avgvol_o2000,sh_opt_optionshort,sh_short_high"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    }

    print("Fetching shorted stocks from finviz...")
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    stocks: list[str] = extract_stocks(soup)

    max_screener_page = max([int(link.text) for link in soup.select(".screener-pages")[:-1]])
    while int(soup.select_one(".screener-pages.is-selected").text) < max_screener_page:
        curr_page = int(soup.select_one(".screener-pages.is-selected").text)
        new_url = f"{url}&r={curr_page * 20 + 1}"
        print("Waiting before requesting next page...")
        time.sleep(0.3)
        res = requests.get(new_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        stocks += extract_stocks(soup)

    print(f"Successfully fetched {len(stocks)} shorted stocks")
    return stocks


def social_media_buzz(stocks: list[str], day=(datetime.now() - timedelta(days=1)), site="x.com"):
    """Returns the number of search results for the given stock on the given day."""
    results_map = {stock: None for stock in stocks}
    driver = uc.Chrome()

    for stock in stocks:
        # Formulate the URL
        query = f"site%3A{site}+{stock}"
        start_date = day.strftime("%m/%d/%Y")
        end_date = (day + timedelta(days=1)).strftime("%m/%d/%Y")
        date_range = f"cdr%3A1%2Ccd_min%3A{quote(start_date, ',')}%2Ccd_max%3A{quote(end_date, ',')}"
        url = f"https://www.google.com/search?q={query}&tbs={date_range}"

        # Navigate
        driver.get(url)

        # Read result stats
        result_stats = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#result-stats")))
        try:
            n_results = int(result_stats.text.split(" ")[1].replace(",", ""))
        except Exception:
            print(f"Failed to fetch results for {stock}")
            print(result_stats)
            n_results = -1
            # time.sleep(1000)  # TODO: Use this to workaround captcha or alert user to solve captcha

        results_map[stock] = n_results

        # Wait before next request
        if len(results_map) < len(stocks):
            time.sleep(10)

    return results_map


# print(social_media_buzz("NNE", datetime.strptime("2025-06-13", "%Y-%m-%d")))
# print(highly_shorted_stocks())
