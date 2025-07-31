import asyncio
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails

from typing import List
import threading
import signal
import time
from datetime import datetime, timedelta
import pytz
import traceback
import math

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496
DEFUALT_CLIENT_ID = 0


class App(EWrapper, EClient):
    class MarketDataTypes:
        LIVE = 1
        FROZEN = 2
        DELAYED = 3
        DELAYED_FROZEN = 4

    # Manage multiple clients
    clients: list["App"] = []
    stop_termination_thread = threading.Event()
    stop_termination_thread.set()

    # Exit on Ctrl+C
    signal.signal(signal.SIGINT, lambda sig, frame: (print("Terminating gracefully..."), exit()))

    def __init__(self):
        EClient.__init__(self, self)
        self._next_req_id = 1
        self.tz = pytz.timezone("US/Eastern")
        self.indices = ["SPX", "NDX", "DJI", "VIX", "VIX3M"]
        self.max_retry_depth = 3

        # Store data wrapper received by request id
        self.data_end_flag = False
        self.request_map = {}
        self.market_data_types = {}
        self.ticks = {}
        self.tick_sizes = {27: {}, 28: {}}
        self.tick_prices = {}
        self.contract_details: dict[int, ContractDetails] = {}
        self.sec_def_opt_params = {}
        self.historical_ticks = {}
        self.historical_data = {}

        # Manage market data feeds
        self.max_feeds = 50
        self.live_feeds = []  # req ids
        self.feed_queue = []
        self.snapshot_timeout = 20
        self.timeouts = {}
        self.tick_list_snapshots = {}

    def run(self):
        try:
            super().run()
        except Exception:
            self.print("An error occurred in the message loop:")
            self.print(traceback.format_exc())

    def connect(self, host=DEFAULT_HOST, port=DEFAULT_PORT, client_id=DEFUALT_CLIENT_ID):
        if client_id in [client.clientId for client in App.clients]:
            self.print("ERROR: Given client ID already exists.")
            return

        # Connect and run
        super().connect(host, port, client_id)
        App.clients.append(self)
        threading.Thread(target=self.run, daemon=True).start()

        # Terminate gracefully by terminating daemons when main thread exits
        def terminate():
            main_thread = threading.main_thread()
            while main_thread.is_alive():
                time.sleep(1)
                if App.stop_termination_thread.is_set():
                    return
            for client in App.clients:
                client.disconnect()  # Terminates threads for EReader and run loop

        if App.stop_termination_thread.is_set():
            App.stop_termination_thread.clear()
            threading.Thread(target=terminate, daemon=True).start()

    def disconnect(self):
        super().disconnect()

        # Terminate termination thread if all clients are disconnected
        if all(not client.isConnected() for client in App.clients):
            App.stop_termination_thread.set()

    def print(self, *args):
        if len(App.clients) > 1:
            print(f"[Client-{self.clientId}]", *args)
        else:
            print(*args)

    def next_req_id(self):
        self._next_req_id += 1
        return self._next_req_id - 1

    def get_tick(self, request_map_key: str, tick_type: int):
        if tick_type not in self.ticks or request_map_key not in self.request_map:
            return None

        req_id = self.request_map[request_map_key]
        return self.ticks[tick_type][req_id] if req_id in self.ticks[tick_type] else None

    def in_RTH(self, dt: datetime):
        # weekday_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        mon_to_fri = dt.weekday() < 5
        after_market_open = dt.hour > 9 or (dt.hour == 9 and dt.minute >= 30)
        before_market_close = dt.hour < 16
        return mon_to_fri and after_market_open and before_market_close

    def timestamp_to_datetime(self, timestamp):
        eastern_tz = pytz.timezone("US/Eastern")
        return datetime.fromtimestamp(timestamp).astimezone(eastern_tz)

    def wait_for_end_flag(self):
        while not self.data_end_flag:
            self.print("Waiting for end flag...")
            time.sleep(1)
        self.data_end_flag = False

    async def async_wait_for_end_flag(self):
        while not self.data_end_flag:
            self.print("Waiting for end flag...")
            await asyncio.sleep(1)
        self.data_end_flag = False

    def make_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.exchange = "CBOE" if symbol in self.indices else "SMART"
        contract.currency = "USD"
        contract.secType = "IND" if symbol in self.indices else "STK"
        return contract

    def make_option_contract(self, symbol, expiry="", strike="", right=""):
        contract = Contract()
        contract.symbol = symbol
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.multiplier = "100"
        contract.secType = "OPT"
        return contract

    def contract_id(self, symbol):
        req_id = self.next_req_id()
        self.reqContractDetails(req_id, self.make_contract(symbol))

        while req_id not in self.contract_details:
            time.sleep(1)

        return self.contract_details[req_id][0].contract.conId

    def parse_tick_13(self, tick):
        try:
            iv = tick[2]
            delta = tick[3]
            price = tick[4]
            gamma = tick[6]
            vega = tick[7]
            theta = tick[8]
            underlying_price = tick[9]
            timestamp = tick[10]
        except Exception:
            self.print(f"Error trying to parse tick: {tick}")
            self.print(traceback.format_exc())
            return None

        return {
            "iv": iv,
            "delta": delta,
            "price": price,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "underlying_price": underlying_price,
            "timestamp": timestamp,
        }

    def greeks(self, option_contracts: List[Contract], allow_cached=False):
        greeks = [None] * len(option_contracts)

        if allow_cached:
            for i, contract in enumerate(option_contracts):
                expiry = contract.lastTradeDateOrContractMonth
                request_map_key = f"{contract.symbol}-{expiry}-{contract.strike}-{contract.right}"
                cached_tick = self.get_tick(request_map_key, 13)
                greeks[i] = self.parse_tick_13(cached_tick) if cached_tick is not None else None

            if None not in greeks:
                return greeks

        self.reqMarketDataType(self.MarketDataTypes.FROZEN)
        contracts = [contract for i, contract in enumerate(option_contracts) if greeks[i] is None]
        req_ids = [self.next_req_id() for contract in contracts]
        start_time = time.time()

        for contract, req_id in zip(option_contracts, req_ids):
            self.reqMktData(req_id, contract, "", True)

        while len(self.live_feeds) > 0:
            self.print("Waiting to receive greeks...")
            time.sleep(1)

        end_time = time.time()
        exec_time = end_time - start_time
        self.print(f"Received {len(req_ids)} greeks in {exec_time:.2f} seconds")

        for req_id in req_ids:
            for i, greek in enumerate(greeks):
                if greek is None:
                    greeks[i] = self.parse_tick_13(self.ticks[13][req_id]) if req_id in self.ticks[13] else {}
                    break

        return greeks

    async def open_interests(self, contracts: List[Contract], retry_depth=0):
        """Returned OIs correspond to given contracts of same index."""
        open_interests = []
        start_time = time.time()
        self.reqMarketDataType(self.MarketDataTypes.LIVE)
        req_ids = [self.next_req_id() for contract in contracts]

        self.print(f"Requesting open interests for {len(contracts)} contracts")
        for contract, req_id in zip(contracts, req_ids):
            self.reqMktData(req_id, contract, "101", True)

        while len(self.live_feeds) > 0:
            self.print("Waiting to receive open interests...")
            await asyncio.sleep(1)

        end_time = time.time()
        exec_time = end_time - start_time
        self.print(f"Received {len(req_ids)} open interests in {exec_time:.2f} seconds")

        for contract, req_id in zip(contracts, req_ids):
            right = contract.right
            if right == "C" and req_id in self.tick_sizes[27]:
                oi = int(self.tick_sizes[27][req_id])
            elif right == "P" and req_id in self.tick_sizes[28]:
                oi = int(self.tick_sizes[28][req_id])
            else:
                oi = None
            open_interests.append(oi)

        # Retry until all open interests are received or max retry depth is reached
        if None in open_interests and retry_depth < self.max_retry_depth:
            self.print(f"Retrying {open_interests.count(None)} missing open interests")
            retry_contracts = [contract for contract, oi in zip(contracts, open_interests) if oi is None]
            retry_open_interests = await self.open_interests(retry_contracts, retry_depth + 1)

            for retry_oi in retry_open_interests:
                for i in range(len(open_interests)):
                    if open_interests[i] is None:
                        open_interests[i] = retry_oi
                        break

        return open_interests

    async def option_contracts(self, symbol: str, expirations: List[str], strike="", right=""):
        req_ids: list[int] = []

        for expiry in expirations:
            req_id = self.next_req_id()
            req_ids.append(req_id)
            contract = self.make_option_contract(symbol, expiry, strike=strike, right=right)

            # Send request and measure time
            start_time = time.time()
            self.reqContractDetails(req_id, contract)
            await self.async_wait_for_end_flag()
            end_time = time.time()
            exec_time = end_time - start_time
            self.print(f"Done collecting options for expiration {expiry} in {exec_time:.2f} seconds")
            self.print(f"Done {len(req_ids)} out of {len(expirations)} expirations")

            # Skip waiting if done all requests
            if len(req_ids) == len(expirations):
                break

            # Detect and counteract throttling
            self.print("Waiting to send next request...")
            if exec_time > 30:
                await asyncio.sleep(48)
            elif exec_time > 20:
                await asyncio.sleep(24)
            elif exec_time > 10:
                await asyncio.sleep(12)
            elif exec_time > 5:
                await asyncio.sleep(6)
            else:
                await asyncio.sleep(3)

        contract_details: list[ContractDetails] = sum([self.contract_details[req_id] for req_id in req_ids], [])
        contracts = [cd.contract for cd in contract_details]
        return contracts

    def option_expirations(self, symbol):
        con_id = self.contract_id(symbol)
        req_id = self.next_req_id()

        # Execute request and measure time
        start_time = time.time()
        self.print(f"Requesting option expirations for {symbol}")
        self.reqSecDefOptParams(req_id, symbol, con_id)
        self.wait_for_end_flag()
        end_time = time.time()
        exec_time = end_time - start_time

        today = datetime.now()
        expirations: dict[str, list[str]] = {
            params[2]: sorted(list(params[4]), key=lambda expiry: datetime.strptime(expiry, "%Y%m%d"))
            for params in self.sec_def_opt_params[req_id]
            if params[0] == "SMART"
        }
        # If market closed, remove today's expiration b/c those options are expired
        if today.hour >= 16:
            expirations = {
                tradingClass: [
                    dt.strftime("%Y%m%d")
                    for dt in [datetime.strptime(e, "%Y%m%d") for e in exps]
                    if not (dt.day == today.day and dt.month == today.month and dt.year == today.year)
                ]
                for tradingClass, exps in expirations.items()
            }

        self.print(f"Received all option expirations for {symbol} in {exec_time:.2f} seconds")
        return expirations

    def get_historical_ticks(self, contract, what_to_show, days):
        end = datetime.now().astimezone(self.tz)
        if end.hour >= 16:
            end = end.replace(hour=16, minute=0, second=0, microsecond=0)
        end_str = end.strftime("%Y%m%d %H:%M:%S US/Eastern")
        start = end - timedelta(days=days)
        all_ticks = []
        overall_start_time = time.time()

        while len(all_ticks) == 0 or start < self.timestamp_to_datetime(all_ticks[0].time):
            req_id = self.next_req_id()
            start_time = time.time()
            self.print(f"Requesting {what_to_show} ticks until {end_str}")
            self.reqHistoricalTicks(req_id, contract, "", end_str, 1000, what_to_show, 1, True, [])
            self.wait_for_req_id(req_id, poll_time=0.2)
            end_time = time.time()
            exec_time = end_time - start_time
            self.print(f"Received {what_to_show} ticks in {exec_time:.2f} seconds")

            ticks = self.data[req_id]
            end = self.timestamp_to_datetime(ticks[0].time)
            end_str = end.strftime("%Y%m%d %H:%M:%S US/Eastern")
            all_ticks = ticks + all_ticks

            # Mitigate throttling
            sleep = 0
            if exec_time > 15:
                sleep = 60
            elif exec_time > 10:
                sleep = 20
            elif exec_time > 5:
                sleep = 10
            elif exec_time > 3:
                sleep = 5
            elif exec_time > 2:
                sleep = 2
            else:
                sleep = 0.3
            self.print(f"Sleeping before next request (anti-throttle: {sleep}s)...")
            time.sleep(sleep)

        overall_end_time = time.time()
        overall_exec_time = overall_end_time - overall_start_time
        start_str = start.strftime("%Y%m%d %H:%M:%S US/Eastern")
        all_ticks = [tick for tick in all_ticks if self.timestamp_to_datetime(tick.time) >= start]
        self.print(f"Received all {len(all_ticks)} {what_to_show} ticks from {start_str} in {overall_exec_time:.2f} seconds")

        return all_ticks

    def get_historical_trades(self, contract, days=1):
        return self.get_historical_ticks(contract, "TRADES", days)

    def get_historical_bid_asks(self, contract, days=1):
        return self.get_historical_ticks(contract, "BID_ASK", days)

    def get_historical_bid_asks_for_trades(self, contract, historical_trade_ticks):
        bid_ask_ticks = {}
        unique_trade_times = sorted(list(set(trade.time for trade in historical_trade_ticks)))
        overall_start_time = time.time()

        # Define chunks of requests to run concurrently
        chunk_size = 4
        chunks = [unique_trade_times[i : i + chunk_size] for i in range(0, len(unique_trade_times), chunk_size)]

        for timestamps in chunks:
            req_ids = []
            start_time = time.time()
            self.print(f"Requesting ticks for chunk: {timestamps}")

            # Request ticks concurrently
            for timestamp in timestamps:
                end = datetime.fromtimestamp(timestamp).astimezone(self.tz) + timedelta(seconds=1)
                end_str = end.strftime("%Y%m%d %H:%M:%S US/Eastern")
                req_id = self.next_req_id()
                self.reqHistoricalTicks(req_id, contract, "", end_str, 1, "BID_ASK", 1, True, [])
                req_ids.append(req_id)

            # Wait for chunk
            while not all(req_id in self.data for req_id in req_ids):
                self.print("Waiting for chunk to complete...")
                time.sleep(1)

            # Measure time
            end_time = time.time()
            exec_time = end_time - start_time
            self.print(f"Received bid-ask chunk ticks in {exec_time:.2f} seconds")

            # Parse received chunk
            for timestamp, req_id in zip(timestamps, req_ids):
                bid_ask_ticks[timestamp] = self.data[req_id]
            self.print(f"Done {len(bid_ask_ticks)} out of {len(unique_trade_times)} ticks")

            # Skip throttle control if done
            if len(bid_ask_ticks) == len(unique_trade_times):
                break

            # Throttle control
            sleep = 0
            if exec_time > 20:
                sleep = 20
            elif exec_time > 10:
                sleep = 10
            elif exec_time > 5:
                sleep = 5
            elif exec_time > 3:
                sleep = 3
            elif exec_time > 1:
                sleep = 2
            else:
                sleep = 1
            self.print(f"Sleeping before next request (anti-throttle: {sleep}s)...")
            time.sleep(sleep)

        # Confirm all timestamps retreived
        self.print(f"Confirm all timestamps received: {all(timestamp in bid_ask_ticks for timestamp in unique_trade_times)}")

        # Measure overall time
        overall_end_time = time.time()
        overall_exec_time = overall_end_time - overall_start_time
        self.print(f"All bid-ask ticks received in {overall_exec_time:.2f} seconds")

        return bid_ask_ticks

    def spot_price(self, symbol, use_RTH=True):
        contract = self.make_contract(symbol)
        req_id = self.next_req_id()
        end = datetime.now().strftime("%Y%m%d %H:%M:%S America/New_York")

        self.reqHistoricalData(req_id, contract, end, "60 S", "1 min", "TRADES", use_RTH=use_RTH)

        while req_id not in self.historical_data:
            self.print("Waiting for spot price...")
            time.sleep(1)

        spot_price: float = self.historical_data[req_id][0].close
        self.print(f"Fetched spot price of {spot_price:.2f}")

        dt_str = self.historical_data[req_id][0].date
        if "US/Central" in dt_str:
            dt = datetime.strptime(self.historical_data[req_id][0].date, "%Y%m%d %H:%M:%S US/Central")
            dt = dt + timedelta(hours=1, minutes=1)  # Convert to US/Eastern
        else:
            dt = datetime.strptime(self.historical_data[req_id][0].date, "%Y%m%d %H:%M:%S US/Eastern")
        return spot_price, dt

    def closes(self, symbol: str, days: int):
        contract = self.make_contract(symbol)
        req_id = self.next_req_id()
        end = datetime.now().strftime("%Y%m%d %H:%M:%S America/New_York")
        duration = f"{days} D" if days <= 365 else f"{math.ceil(days / 248)} Y"  # 248 is low-end estimate of trading days in year

        self.reqHistoricalData(req_id, contract, end, duration, "1 day", "TRADES", use_RTH=True)

        while req_id not in self.historical_data:
            self.print(f"Waiting for closes for {symbol}...")
            time.sleep(1)

        closes = {bar.date: bar.close for bar in self.historical_data[req_id][-days:]}
        return closes

    ##
    ## WRAPPER (METHODS RUN IN MESSAGE LOOP THREAD)
    ##

    def contractDetails(self, reqId, contractDetails):
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []

        self.contract_details[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        self.data_end_flag = True

    def historicalTicksBidAsk(self, reqId, ticks, done):
        # for tick in ticks:
        #     self.print(f"{(tick.time, tick.priceBid, tick.priceAsk)}")
        self.historical_ticks[reqId] = ticks

    def historicalTicksLast(self, reqId, ticks, done):
        # for tick in ticks:
        #     self.print(f"{(tick.time, tick.price)}")
        self.historical_ticks[reqId] = ticks

    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []

        self.historical_data[reqId].append(bar)

    def marketDataType(self, reqId, marketDataType):
        self.market_data_types[reqId] = marketDataType

    def securityDefinitionOptionParameter(self, reqId, *args):
        if reqId not in self.sec_def_opt_params:
            self.sec_def_opt_params[reqId] = []

        self.sec_def_opt_params[reqId].append(args)

    def securityDefinitionOptionParameterEnd(self, reqId):
        self.data_end_flag = True

    def tickOptionComputation(self, reqId, *args):
        tick_type = args[0]

        if tick_type not in self.ticks:
            self.ticks[tick_type] = {}

        self.ticks[tick_type][reqId] = (*args, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def tickSize(self, reqId, tickType, size):
        if tickType not in self.tick_sizes:
            self.tick_sizes[tickType] = {}

        self.tick_sizes[tickType][reqId] = size

        if reqId in self.tick_list_snapshots and self.tick_list_snapshots[reqId] == "101":
            call_oi_recvd = 27 in self.tick_sizes and reqId in self.tick_sizes[27]
            put_oi_recvd = 28 in self.tick_sizes and reqId in self.tick_sizes[28]
            greeks_recvd = 13 in self.ticks and reqId in self.ticks[13]
            in_RTH = self.in_RTH(datetime.now())
            if call_oi_recvd and put_oi_recvd and (not in_RTH or greeks_recvd):
                if reqId in self.live_feeds:
                    self.cancelMktData(reqId)

    def tickPrice(self, reqId, tickType, price, attrib):
        self.tick_prices[reqId] = (tickType, price, attrib)

    def tickSnapshotEnd(self, reqId):
        if reqId in self.timeouts:
            self.timeouts.pop(reqId)

        if reqId not in self.live_feeds:
            self.print(f"WARNING: Tick snapshot end received for {reqId} not in live feeds list. ")
            return

        self.live_feeds.remove(reqId)
        self.dequeue_feed_queue()

        self.print(f"Done req_id {reqId}")

    ##
    ## CLIENT
    ##

    def _wait_until_connected(self):
        # while self.conn is None or not self.conn.isConnected():
        #     time.sleep(0.1)
        # self.print(f"Is connected: {self.conn.isConnected()}")
        pass

    def reqContractDetails(self, req_id, contract):
        self._wait_until_connected()
        self.data_end_flag = False
        return super().reqContractDetails(req_id, contract)

    def reqMarketDataType(self, *args, **kwargs):
        self._wait_until_connected()
        return super().reqMarketDataType(*args, **kwargs)

    def timeout_snapshots(self):
        self.timeout_thread = threading.Event()

        while len(self.timeouts) > 0:
            timeouts = self.timeouts.copy()
            for req_id, timeout in timeouts.items():
                if time.time() > timeout:
                    self.print(f"Timing out req_id {req_id}")
                    self.timeout_thread.clear()
                    self.cancelMktData(req_id)
                    self.timeout_thread.wait()

            time.sleep(0.5)

    def reqMktData(self, req_id: int, contract: Contract, tick_list: str, snapshot: bool):
        self._wait_until_connected()

        if contract.secType == "OPT":
            expiry = contract.lastTradeDateOrContractMonth
            request_map_key = f"{contract.symbol}-{expiry}-{contract.strike}-{contract.right}"
        else:
            request_map_key = contract.symbol
        self.request_map[request_map_key] = req_id

        if len(self.live_feeds) < self.max_feeds:
            self.live_feeds.append(req_id)

            if snapshot:
                self.timeouts[req_id] = time.time() + self.snapshot_timeout

                if tick_list != "":
                    snapshot = False
                    self.tick_list_snapshots[req_id] = tick_list

            super().reqMktData(req_id, contract, tick_list, snapshot, False, [])
        else:
            self.feed_queue.append(lambda: self.reqMktData(req_id, contract, tick_list, snapshot))

        if f"[Client-{self.clientId}] {self.timeout_snapshots.__name__}" not in [thread.name for thread in threading.enumerate()]:
            threading.Thread(
                target=self.timeout_snapshots, daemon=True, name=f"[Client-{self.clientId}] {self.timeout_snapshots.__name__}"
            ).start()

    def dequeue_feed_queue(self):
        if len(self.feed_queue) > 0:
            next_feed = self.feed_queue[0]
            self.feed_queue = self.feed_queue[1:]
            next_feed()

    def cancelMktData(self, req_id):
        super().cancelMktData(req_id)

        if req_id in self.timeouts:
            self.timeouts.pop(req_id)

        if req_id not in self.live_feeds:
            self.print(f"WARNING: Requested to cancel req_id {req_id} not in live feeds list. ")
            return

        self.live_feeds.remove(req_id)
        self.dequeue_feed_queue()

        self.print(f"Done req_id {req_id}")
        self.timeout_thread.set()

    def reqSecDefOptParams(self, req_id: int, symbol: str, con_id: int):
        sec_type = "IND" if symbol in self.indices else "STK"
        self.data_end_flag = False
        return super().reqSecDefOptParams(req_id, symbol, "", sec_type, con_id)

    def reqHistoricalData(self, req_id, contract, end, duration, bar_size, what_to_show, use_RTH=True):
        use_RTH = int(use_RTH)
        return super().reqHistoricalData(req_id, contract, end, duration, bar_size, what_to_show, use_RTH, 1, False, [])
