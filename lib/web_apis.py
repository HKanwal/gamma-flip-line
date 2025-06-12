import requests


def risk_free_rate():
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
        return risk_free_rate
    else:
        raise Exception(f"Risk-free rate fetch failed with status code {res.status_code}.")
