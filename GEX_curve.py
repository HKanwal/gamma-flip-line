import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from datetime import datetime
import os

import lib.ibkr as ibkr
import lib.mibian as mibian
import lib.web_apis as web_apis

# Must not have dividends and must have European style contracts for BS model to be accurate
SYMBOL = "SPX"

total_runtime_start = time.time()
app = ibkr.App()
app.connect()

# Collect expirations
expirations = app.option_expirations(SYMBOL)
monthly_expirations = expirations[SYMBOL]
weekly_expirations = [exp for exp in expirations[f"{SYMBOL}W"] if exp not in monthly_expirations]
sample_expirations = [
    weekly_expirations[0],
    weekly_expirations[1],
    monthly_expirations[0],
    monthly_expirations[1],
]
sample_expirations.sort(key=lambda exp: datetime.strptime(exp, "%Y%m%d"))
print("Sampling expirations:", sample_expirations)

# Collect options contracts
contracts: list[ibkr.Contract] = app.option_contracts(SYMBOL, sample_expirations)

# # Get spot price and spot datetime
spot_price, dt = app.spot_price(SYMBOL)
print(f"Using spot price of {spot_price}")

# Collect open interests and IVs (greeks)
n_downside_strikes = 300
n_upside_strikes = 100
# n_downside_strikes = n_upside_strikes = 10  # For testing purposes only
contracts = list(filter(lambda c: c.tradingClass in [SYMBOL, f"{SYMBOL}W"], contracts))
strikes = sorted(list(set([c.strike for c in contracts])))
strikes = (
    list(filter(lambda strike: strike < spot_price, strikes))[-n_downside_strikes:]
    + list(filter(lambda strike: strike > spot_price, strikes))[:n_upside_strikes]
)
contracts = [contract for contract in contracts if contract.strike in strikes]
strike_range = (strikes[0], strikes[-1])
print(f"Sampling strike range of {strike_range}")

open_interests = app.open_interests(contracts)
greeks = app.greeks(contracts, allow_cached=True)
app.disconnect()

# Compile data into a dataframe
data = {
    "Expiry": [],
    "Strike": [],
    "Right": [],
    "IV": [],
    "Gamma": [],
    "Open Interest": [],
}
for contract, oi, greek in zip(contracts, open_interests, greeks):
    data["Expiry"].append(contract.lastTradeDateOrContractMonth)
    data["Strike"].append(contract.strike)
    data["Right"].append(contract.right)
    data["Open Interest"].append(oi)
    data["IV"].append(greek["iv"] if "iv" in greek else "NaN")
    data["Gamma"].append(greek["gamma"] if "iv" in greek else "NaN")
data_df = pd.DataFrame(data)
os.makedirs("data", exist_ok=True)
data_df.to_csv("data/gamma_flip_line_cache.csv", index=False)  # For debugging purposes, reuse data that causes crashes

# Prep data
# data_df = pd.read_csv("data/gamma_flip_line_cache.csv")
data_df["IV"] = data_df["IV"] * 100
data_df["Expiry"] = pd.to_datetime(data_df["Expiry"], format="%Y%m%d") + pd.Timedelta(hours=16)
data_df["DTE"] = (data_df["Expiry"] - dt).dt.total_seconds() / (60 * 60 * 24)


def net_gamma(price, data_df, risk_free_rate):
    """Calculate net gamma using Black-Scholes across all options (rows) in given dataframe for given underlying price."""

    def model_gamma(row):
        try:
            bs = mibian.BS([price, row["Strike"], risk_free_rate, row["DTE"]], volatility=row["IV"])
        except Exception as e:
            print(row)
            raise e
        return bs._gamma()

    data_df["Model Gamma"] = data_df.apply(model_gamma, axis=1)
    data_df["Gamma Contribution"] = (
        data_df["Model Gamma"] * data_df["Open Interest"] * ((data_df["Right"] == "C").astype(int) * 2 - 1)
    )
    return data_df["Gamma Contribution"].sum() * 100


def newton_method(fn, x1, y_threshhold, d=1e-6):
    """Approximates the x-intercept of given function starting from given initial guess."""
    inst_rise = fn(x1 + d) - fn(x1 - d)
    inst_run = 2 * d
    m = inst_rise / inst_run
    b = fn(x1) - m * x1
    zero = -b / m

    if abs(fn(zero)) < y_threshhold:
        return zero
    else:
        return newton_method(fn, zero, y_threshhold)


# Calculate gamma flip line
risk_free_rate = web_apis.risk_free_rate()
print("Calculating gamma flip line...")
start_time = time.time()
gamma_flip_price = newton_method(lambda p: net_gamma(p, data_df, risk_free_rate), spot_price, 10)
end_time = time.time()
exec_time = end_time - start_time
print(f"Gamma Flip Price of {gamma_flip_price:.2f} calculated in {exec_time:.2f} seconds")

# Prep data for plotting
x_lim = (int(min(gamma_flip_price, spot_price)) - 500, int(max(gamma_flip_price, spot_price)) + 500)
x_mid = int((x_lim[0] + x_lim[1]) / 2)
xs = (
    list(range(x_lim[0], x_mid - 200, 50))
    + list(range(x_mid - 200, x_mid + 200, 10))
    + list(range(x_mid + 200, x_lim[1], 50))
)
print("Computing curve...")
start_time = time.time()
ys = [net_gamma(x, data_df, risk_free_rate) for x in xs]
end_time = time.time()
exec_time = end_time - start_time
print(f"Curve computed in {exec_time:.2f} seconds")

total_runtime_end = time.time()
total_runtime = total_runtime_end - total_runtime_start
print(f"Total runtime of {total_runtime:.2f} seconds")

# Create plot
title = f"{SYMBOL} Gamma Exposure (GEX) for {dt.strftime('%b %d, %Y')}"
fig = plt.figure(figsize=(12, 6), facecolor="beige")
fig.canvas.manager.set_window_title(title)
gs = GridSpec(2, 1, height_ratios=[59, 1])
ax1 = fig.add_subplot(gs[0], facecolor="beige")
ax1.plot(xs, ys, linestyle="-", color="green", label="Net Gamma Curve")
ax1.set_ylabel("Net Gamma")
ax1.set_xlabel("Underlying Price", labelpad=20)
ax1.grid(True, which="major", linestyle="--", alpha=0.7)
ax1.grid(True, which="minor", linestyle=":", alpha=0.4)
ax1.minorticks_on()

# Plot lines
ax1.axhline(y=0, color="black", linewidth=1, linestyle="--")
ax1.axvline(x=spot_price, color="blue", linewidth=1, label="Spot Price")
ax1.axvline(x=gamma_flip_price, color="red", linewidth=1, linestyle="--", label="Gamma Flip Line")

# Create ticks for vertical lines
special_ticks = [(gamma_flip_price, "red"), (spot_price, "blue")]
ax2 = ax1.secondary_xaxis("bottom")
ax2.spines["bottom"].set_visible(False)
ticks = [tick[0] for tick in special_ticks]
ax2.set_xticks(ticks)
# ax2.set_xticklabels([f"{tick:.2f}" for tick in ticks])  # Causes overlap
ax2.set_xticklabels([f"{ticks[0]:.2f}", ""])
ax2.tick_params(axis="x", length=15, width=1)

# Set individual colors for each tick label
for ticklabel, tick in zip(ax2.get_xticklabels(), [tick for tick in special_ticks]):
    ticklabel.set_color(tick[1])

# Set individual colors for each tick
for tick, color in zip(ax2.xaxis.get_major_ticks(), [tick[1] for tick in special_ticks]):
    tick.tick1line.set_markeredgecolor(color)
    # tick.tick1line.set_markersize(15)
    # tick.tick1line.set_markeredgewidth(1)

# Sampled expirations
fmt_expirations = ", ".join(f"{datetime.strptime(exp, '%Y%m%d').strftime('%b %d %Y')}" for exp in sample_expirations)
ax1.text(
    0.99,
    0.02,
    f"Sampled Expirations:\n{fmt_expirations}\n\nSampled Strike Range:\n{strike_range}",
    transform=ax1.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8", boxstyle="round,pad=0.5"),
)

# Sources
plt.figtext(
    0.015,
    0.025,
    "Sources: Interactive Brokers, hkanwal.com",
    ha="left",
    va="bottom",
    fontsize=10,
    color="black",
    alpha=0.6,
)

# Show plot
ax1.set_title(title, fontsize=16, fontweight="bold", pad=10)
ax1.legend(loc="upper left")
plt.tight_layout()
plt.show()


# # Collect estimates of dealer positioning
# oi_cutoff = 500
# gamma_cutoff = 2e-3
# data_df = pd.read_csv("data/gamma_flip_line_cache.csv")
# # truncated_df = data_df.drop(data_df.index[16:])
# # truncated_df = truncated_df.drop(truncated_df.index[:15])
# truncated_df = data_df
# truncated_df["i"] = truncated_df.index
# total_rows = len(truncated_df)


# def estimate_dealer_short_percentage_using_order_flow(row):
#     """Estimate dealer's short percentage of open interest for given option."""
#     expiry = row["Expiry"]
#     strike = row["Strike"]
#     right = row["Right"]
#     id_str = f"({expiry}, {strike}, {right})"

#     # Ignore row if contribution will be negligible
#     if row["Open Interest"] < oi_cutoff:
#         print(f"Ignored option below OI cutoff: {id_str}")
#         return 0
#     elif row["Gamma"] < gamma_cutoff:
#         print(f"Ignored option below gamma cutoff: {id_str}")
#         return 0

#     contract = app.make_options_contract(SYMBOL, expiry, strike, right)
#     print(f"Begin estimation for {id_str}")

#     # Collect order flow data
#     sample_hours = 2
#     sample_days = sample_hours / 24
#     historical_trades = app.get_historical_trades(contract, sample_days)
#     if len(historical_trades) == 0:
#         print(f"No historical trades for {id_str}")
#         return 0
#     historical_bid_asks = app.get_historical_bid_asks_for_trades(contract, historical_trades)

#     # Parse bid-ask ticks
#     bid_ask_map = {}
#     for timestamp in historical_bid_asks:
#         if timestamp not in bid_ask_map:
#             bid_ask_map[timestamp] = {"bids": [], "asks": []}

#         for tick in historical_bid_asks[timestamp]:
#             bid_ask_map[timestamp]["bids"].append(tick.priceBid)
#             bid_ask_map[timestamp]["asks"].append(tick.priceAsk)

#     def trade_direction(price, bids, asks):
#         """Determine direction of trade. Dealer buys at bid and sells at ask."""
#         is_ambiguous = price in bids and price in asks
#         between_bids = price > min(bids) and price < max(bids)
#         below_asks = price < min(asks)
#         between_asks = price < max(asks) and price > min(asks)
#         above_bids = price > max(bids)
#         between_spread = price > max(bids) and price < min(asks)
#         below_bid = price < min(bids)
#         above_ask = price > max(asks)

#         if is_ambiguous:
#             # TODO: better ways to handle ambiguity?
#             frequency_greater_in_bids = bids.count(price) > asks.count(price)
#             return "dealer long" if frequency_greater_in_bids else "dealer short"
#         elif price in bids or below_bid or (between_bids and below_asks):
#             return "dealer long"
#         elif price in asks or above_ask or (between_asks and above_bids):
#             return "dealer short"
#         elif between_bids or between_asks:
#             n_bids_below = sum(1 for bid in bids if price < bid)
#             n_asks_above = sum(1 for ask in asks if price > ask)
#             return "dealer long" if n_bids_below > n_asks_above else "dealer short"
#         elif between_spread:
#             closer_to_bid = price - max(bids) < min(asks) - price
#             return "dealer long" if closer_to_bid else "dealer short"
#         else:
#             print("ERROR: This message should not be seen. Something went wrong in determining trade direction.")
#             print(f"Undermined trade direction for {id_str}. Price: {price}, Bids: {bids}, Asks: {asks}")
#             return -1

#     # Count dealer long and short trades
#     dealer_long_trades = 0
#     dealer_short_trades = 0
#     for trade in historical_trades:
#         t = trade.time
#         if t not in bid_ask_map.keys():
#             if len(bid_ask_map) == 0:
#                 print(f"ERROR: Somehow missing all bid-asks for {id_str}.")
#                 return -1
#             t = min(bid_ask_map.keys(), key=lambda k: abs(k - t))

#         bids = bid_ask_map[t]["bids"]
#         asks = bid_ask_map[t]["asks"]

#         direction = trade_direction(trade.price, bids, asks)
#         if direction == "dealer long":
#             dealer_long_trades += trade.size
#         elif direction == "dealer short":
#             dealer_short_trades += trade.size

#     total_trades = dealer_short_trades + dealer_long_trades
#     if total_trades == 0:
#         print("ERROR: This message should not be seen. Total trades should've been equal to # of trade ticks.")
#         return 0
#     print(f"Done estimation for row {row['i']} of {total_rows - 1}")
#     return dealer_short_trades / (dealer_short_trades + dealer_long_trades)


# print(f"Beginning estimation for {total_rows} rows")
# truncated_df["Dealer Short Percentage of OI"] = truncated_df.apply(
#     estimate_dealer_short_percentage_using_order_flow, axis=1
# )
# truncated_df.to_csv("data/gamma_flip_line_cache_2.csv", index=False)
# pd.set_option("display.max_rows", None)
# print(truncated_df)
