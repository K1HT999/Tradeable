import asyncio
import logging
import datetime
import os
import pandas as pd
import numpy as np
import nest_asyncio
from typing import List, Dict
import pytz

# Alpaca SDK (Newer Async Version)
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live.stock import StockDataStream, DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# Patch asyncio for Jupyter/Some IDE environments
nest_asyncio.apply()

# -----------------------------
# CONFIGURATION
# -----------------------------
class Config:
    # ------------------------------------------------------------------
    # SECURITY: Keys are loaded from Environment Variables
    # ------------------------------------------------------------------
    ALPACA_API_KEY = 'INSERT_KEY_HERE'
    ALPACA_SECRET_KEY = 'INSERT_SECRET_HERE' 

    # System Settings
    NY = pytz.timezone("America/New_York")
    LOG_FILE = "execution_engine.log"
    CALC_WINDOW = 50  # Optimization: Only keep 50 bars in memory for O(1) calc speed
    
    # Universe
    TICKERS = ['AMZN', 'MSFT', 'NVDA', 'AAPL']
    
    # Parameters (Redacted/Placeholder)
    RSI_PERIOD = 14
    BB_WINDOW = 20
    BB_STD = 2
    
    # Risk Management
    STOP_LOSS_PCT = 0.005
    TAKE_PROFIT_PCT = 0.015
    MAX_RISK_PER_TRADE = 0.01 # 1% of equity
    MAX_POSITION_PCT = 0.10   # Max 10% in one stock
    
    # Execution Flags
    EXECUTE_REAL_ORDERS = True 
    LIQUIDATION_TIME = datetime.time(15, 30)
    LAST_ENTRY_TIME = datetime.time(15, 00)


# Setup logging - Reduced verbosity for high-frequency performance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(Config.LOG_FILE)]
)
logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self):
        # Initialize Async Trading Client
        self.trading_client = TradingClient(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY, paper=True)
        
        # In-Memory Data Structures (No Database for speed)
        self.dfs: Dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in Config.TICKERS}
        
        # State Tracking
        self.open_positions = {t: False for t in Config.TICKERS}
        self.position_data = {t: {} for t in Config.TICKERS} 
        self.cached_equity = 100000.0 
        
        logger.info("Live Execution Engine initialized")

    async def start_background_tasks(self):
        """
        Start non-blocking maintenance tasks:
        1. Equity Poller (For position sizing)
        2. Position Syncer (Drift correction)
        """
        asyncio.create_task(self._equity_poller())
        asyncio.create_task(self._position_syncer())

    async def _equity_poller(self):
        """Async polling of account equity (every 10s) to avoid API rate limits."""
        while True:
            try:
                acct = await asyncio.to_thread(self.trading_client.get_account)
                self.cached_equity = float(acct.equity)
            except Exception as e:
                logger.error(f"Equity Poll Error: {e}")
            await asyncio.sleep(10)

    async def _position_syncer(self):
        """Failsafe: Reconcile local state with server state every 60s."""
        while True:
            try:
                positions = await asyncio.to_thread(self.trading_client.get_all_positions)
                server_positions = {p.symbol for p in positions}
                
                for ticker in Config.TICKERS:
                    is_open = ticker in server_positions
                    if is_open and not self.open_positions[ticker]:
                        logger.warning(f"State Mismatch: Found open position for {ticker} on server.")
                        self.open_positions[ticker] = True
                    elif not is_open and self.open_positions[ticker]:
                        self.open_positions[ticker] = False
            except Exception as e:
                logger.error(f"Position Sync Error: {e}")
            await asyncio.sleep(60)

    def calculate_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PERFORMANCE OPTIMIZATION:
        Instead of recalculating the entire history, we slice only the 
        last N bars required for the longest indicator window.
        """
        if len(df) < Config.CALC_WINDOW:
            return df

        # Create a view of the tail to save memory/time
        subset = df.iloc[-Config.CALC_WINDOW:].copy()

        # Vectorized Calculations (Pandas/C-backend)
        subset['BB_MID'] = subset['close'].rolling(Config.BB_WINDOW).mean()
        subset['BB_STD'] = subset['close'].rolling(Config.BB_WINDOW).std()
        subset['BB_DN'] = subset['BB_MID'] - Config.BB_STD * subset['BB_STD']

        # RSI Calculation
        delta = subset['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(Config.RSI_PERIOD).mean()
        avg_loss = loss.rolling(Config.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        subset['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic
        # subset['%K'] = ... (Redacted for brevity in live engine)

        # Efficiently update the main dataframe in O(1)
        last_idx = df.index[-1]
        cols_to_update = ['RSI', 'BB_DN']
        
        # Check if columns exist, if not create them
        for col in cols_to_update:
            if col not in df.columns:
                df[col] = np.nan
                
        df.loc[last_idx, cols_to_update] = subset.iloc[-1][cols_to_update]
        
        return df

    def check_signal(self, ticker: str, current_time_ny) -> bool:
        """
        Evaluates trading logic.
        NOTE: Proprietary alpha logic has been redacted for the public repository.
        """
        df = self.dfs[ticker]
        if len(df) < 20: return False
        
        row = df.iloc[-1]
        
        # Data Integrity Check
        if pd.isna(row['RSI']): return False

        # Time Filter
        t = current_time_ny.time()
        if not (datetime.time(9, 30) <= t <= Config.LAST_ENTRY_TIME): return False

        # ---------------------------------------------------
        # STRATEGY LOGIC (REDACTED)
        # ---------------------------------------------------
        # Placeholder logic:
        # signal = (row['RSI'] < 30) and (row['close'] < row['BB_DN'])
        
        signal = False # Default to False for safety in public code
        
        if signal:
            logger.info(f"SIGNAL DETECTED: {ticker}")
            return True
            
        return False

    async def execute_entry(self, ticker: str, price: float, time_ny):
        """
        Executes an OCO (One-Cancels-Other) Bracket Order.
        Async execution ensures we don't block the data stream while waiting for API response.
        """
        if self.open_positions[ticker]: return

        # Dynamic Position Sizing
        equity = self.cached_equity
        stop_price = round(price * (1 - Config.STOP_LOSS_PCT), 2)
        take_profit = round(price * (1 + Config.TAKE_PROFIT_PCT), 2)
        
        risk_per_share = price - stop_price
        if risk_per_share <= 0: return
        
        risk_amt = equity * Config.MAX_RISK_PER_TRADE
        shares = int(risk_amt / risk_per_share)
        
        # Apply Position Limits
        max_shares = int((equity * Config.MAX_POSITION_PCT) / price)
        shares = min(shares, max_shares)
        
        if shares < 1: return

        logger.info(f"EXECUTING {ticker} | Qty:{shares} | @{price}")

        # Submit Order (Non-blocking)
        asyncio.create_task(self._submit_bracket(ticker, shares, take_profit, stop_price, time_ny, price))

    async def _submit_bracket(self, ticker, shares, tp, sl, time_ny, entry_price):
        try:
            if Config.EXECUTE_REAL_ORDERS:
                req = LimitOrderRequest(
                    symbol=ticker,
                    qty=shares,
                    limit_price = entry_price,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl)
                )
                await asyncio.to_thread(self.trading_client.submit_order, order_data=req)
                
            # Optimistic Local State Update (Reduces latency for next check)
            self.open_positions[ticker] = True
            
        except Exception as e:
            logger.error(f"Order Execution Failed {ticker}: {e}")
            self.open_positions[ticker] = False

    async def manage_positions(self, ticker: str, current_row, time_ny):
        """
        Secondary Exit Logic:
        Handles Time-based stops and Indicator-based exits 
        that the OCO Bracket cannot handle.
        """
        # EOD Liquidation
        if time_ny.time() >= Config.LIQUIDATION_TIME:
             logger.info(f"EOD LIQUIDATION TRIGGERED: {ticker}")
             await self.close_position(ticker)

    async def close_position(self, ticker):
        try:
            self.open_positions[ticker] = False
            if Config.EXECUTE_REAL_ORDERS:
                await asyncio.to_thread(self.trading_client.close_position, ticker)
        except Exception as e:
            logger.error(f"Close Error {ticker}: {e}")

    async def on_bar(self, bar):
        """
        Websocket Callback. 
        Runs every time a minute bar closes (or trade updates).
        """
        ticker = bar.symbol
        
        # 1. Ingest Data (Fast TZ conversion)
        ts_ny = pd.to_datetime(bar.timestamp).astimezone(Config.NY)
        
        new_row = pd.DataFrame([{
            'open': bar.open, 'high': bar.high, 'low': bar.low, 
            'close': bar.close, 'volume': bar.volume,
        }], index=[ts_ny])
        
        # 2. Update DataFrame (Rolling window)
        df = self.dfs[ticker]
        df = pd.concat([df, new_row]).iloc[-Config.CALC_WINDOW:] 
        self.dfs[ticker] = df
        
        # 3. Calculate Indicators (Vectorized)
        self.calculate_indicators_vectorized(df)
        
        # 4. Strategy Dispatch
        if self.open_positions[ticker]:
            await self.manage_positions(ticker, df.iloc[-1], ts_ny)
        else:
            if self.check_signal(ticker, ts_ny):
                await self.execute_entry(ticker, bar.close, ts_ny)

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    # Setup Async Loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    engine = TradingEngine()
    
    logger.info("Initializing Historical Warmup...")
    
    # Historical Warmup (Load previous data to prime indicators)
    history_client = StockHistoricalDataClient(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY)
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    start_dt = end_dt - datetime.timedelta(days=2)
    
    for ticker in Config.TICKERS:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.IEX 
            )
            bars = history_client.get_stock_bars(req)
            
            if not bars.data:
                continue

            # Process Historical Data
            df = bars.df.loc[ticker].copy()
            df.index = df.index.tz_convert(Config.NY)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.tail(Config.CALC_WINDOW)
            
            engine.dfs[ticker] = engine.calculate_indicators_vectorized(df)
            logger.info(f"Warmup Complete: {ticker} ({len(df)} bars)")
            
        except Exception as e:
            logger.error(f"Warmup failed for {ticker}: {e}")

    # Start Background Tasks
    loop.create_task(engine.start_background_tasks())

    # Start Websocket Stream
    stream_client = StockDataStream(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY, feed=DataFeed.IEX)
    stream_client.subscribe_bars(engine.on_bar, *Config.TICKERS)
    
    logger.info(f"ENGINE LIVE: Streaming {len(Config.TICKERS)} Tickers")
    
    try:
        stream_client.run()
    except KeyboardInterrupt:
        logger.info("Engine stopped by user")
    except Exception as e:
        logger.critical(f"Fatal Engine Error: {e}")

if __name__ == "__main__":
    main()