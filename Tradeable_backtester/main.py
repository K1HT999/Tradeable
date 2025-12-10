
import os
import warnings
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import time
import datetime
from datetime import datetime
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product
from ta import add_all_ta_features

import uuid

warnings.filterwarnings('ignore')

        
# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
class Config:

    mag_7 = ['AMZN','MSFT','AAPL','NVDA','META','GOOG','TSLA']
        
    # Data parameters
    TICKERS =mag_7

    START = '2024-10-03'
    END = '2025-11-21'
    INTERVAL = '1Min'
    CHUNK_DAYS = 7

    # Technical indicators
    RSI_THRESH = 1
    STOCH_THRESH = 1
    AD_LOOKBACK = 1
    BB_WINDOW = 1
    BB_STD = 1
    RSI_PERIOD = 1
    STOCH_PERIOD = 1
    SMA_PERIOD = 1
    vwap_period = 1
    ATR_PERIOD = 1
    MFI_Period = 1  # Standard period, adjustable if needed
    MFI_THRESHOLD_LOW = 1  # Oversold threshold
    MFI_THRESHOLD_HIGH = 1  # Overbought threshold

    SLIPPAGE_ENABLED = True
    SLIPPAGE_ATR_MULTIPLIER = 0.1

    COMMISSIONS_ENABLED = False
    COMMISSION_PER_SHARE = 0.00002
    MIN_COMMISSION_PER_ORDER = 0.0
    MAX_DAILY_DRAWDOWN_PCT = 0.02

    # Risk management
    STOP_LOSS_PCT = 0.007     #
    TAKE_PROFIT_PCT = 0.02  #  
    INITIAL_EQUITY = 25000 #PDT min
    LEVERAGE = 1.0            
    MAX_RISK_PER_TRADE =0.25 
    MAX_POSITION_PCT = 0.70   
    TRAILING_STOP_ENABLED = False

    # Trading hours (ET)
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    LIQUIDATION_TIME = "15:30"  # 3:30 PM ET
    LAST_ENTRY_TIME = "14:00"   # No new entries after X:XX PM

    TIME_STOP_ENABLED = True
    MAX_HOLD_MINUTES_NO_PROFIT = 1  # Exit after X mins
    
    # Additional filters
    MIN_VOLUME = 1  # Minimum volume for entry

    #no lunchtime trades
    MIDDAY_FILTER_ENABLED = True
    MIDDAY_AVOID_START = "11:30"
    MIDDAY_AVOID_END = "12:30"

    THESIS_INVALIDATION_ENABLED = False

# ----------------------------------------------------------------------
# DATA HANDLING
# ----------------------------------------------------------------------
def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that fall inside regular market hours."""
    return df.between_time(Config.MARKET_OPEN, Config.MARKET_CLOSE)


def download_intraday_chunks(
    api: tradeapi.REST,
    ticker: str,
    start: str,
    end: str,
    interval: str = '1Min',
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Download intraday minute data in chunks with correct timezone and market-hours filtering.
    """

    start_dt = pd.to_datetime(start).tz_localize("America/New_York")
    end_dt = pd.to_datetime(end).tz_localize("America/New_York")

    cur = start_dt
    parts = []

    # Define US market hours (Eastern Time)
    market_open = time(9, 30)
    market_close = time(16, 0)

    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_dt)

        try:
            bars = api.get_bars(
            ticker,
            tradeapi.TimeFrame.Minute,
            start=cur.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=nxt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            ).df
            if bars.empty:
                cur = nxt + pd.Timedelta(seconds=1)
                continue


            # Alpaca always returns UTC timestamps — convert to NY time first

            bars.index = bars.index.tz_convert("America/New_York")


            # Filter by regular market hours in ET

            bars = bars.between_time(market_open, market_close, inclusive="both")

            # Ensure chronological order + no duplicates
            bars = bars[~bars.index.duplicated(keep="first")].sort_index()
            parts.append(bars)

        except Exception as e:
            print(f" Error downloading {ticker} from {cur} to {nxt}: {e}")

        cur = nxt + pd.Timedelta(seconds=1)

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts)
    # Clip to date range (still ET)
    full = full.loc[(full.index >= start_dt) & (full.index <= end_dt)]

    return full


# ----------------------------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------------------------
def calculate_bollinger_bands(df: pd.DataFrame, window: int, std_dev: int) -> pd.DataFrame:
    df['BB_MID'] = df['close'].rolling(window).mean()
    df['BB_STD'] = df['close'].rolling(window).std()
    df['BB_UP'] = df['BB_MID'] + std_dev * df['BB_STD']
    df['BB_DN'] = df['BB_MID'] - std_dev * df['BB_STD']
    return df

def calculate_bullish_divergence(df: pd.DataFrame, lookback: int = 25) -> pd.Series:
    """
    Finds bullish divergence between price (lower lows) and RSI (higher lows).
    
    Args:
        df: DataFrame with 'low' and 'RSI' columns.
        lookback: How many bars back to check for divergence.
        
    Returns:
        A boolean Series, True where divergence is detected.
    """
    # Use shift(1) to ensure we are comparing the current low to previous rolling lows
    price_low_lookback = df['low'].shift(1).rolling(window=lookback, min_periods=3).min()
    rsi_low_lookback = df['RSI'].shift(1).rolling(window=lookback, min_periods=3).min()

    # Condition 1: The current bar's low is a new low compared to the lookback period
    is_new_price_low = df['low'] < price_low_lookback
    
    # Condition 2: But the RSI at this point is HIGHER than its low in the lookback period
    is_higher_rsi_low = df['RSI'] > rsi_low_lookback
    
    # Divergence is when both are true
    divergence = is_new_price_low & is_higher_rsi_low
    return divergence

def calculate_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_stochastic(df: pd.DataFrame, period: int) -> pd.DataFrame:
    low_n = df['low'].rolling(period).min()
    high_n = df['high'].rolling(period).max()
    df['%K'] = 100 * (df['close'] - low_n) / (high_n - low_n).replace(0, np.nan)
    df['%K_slope'] = df['%K'].diff()
    df['%K_slope_3'] = df['%K'].diff(3)
    df['%D'] = df['%K'].rolling(3).mean()
    return df

def calculate_sma(df:pd.DataFrame, period: int) -> pd.DataFrame:
    df['SMA'] = df['close'].rolling(period).mean()
    return df


def calculate_accumulation_distribution(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low']).replace(0, np.nan)
    mfv = mfm * df['volume']
    df['AD'] = mfv.fillna(0).cumsum()
    df['AD_LOW'] = df['AD'].rolling(lookback).min()
    return df


def calculate_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    return df

def calculate_EMA(df:pd.DataFrame) -> pd.DataFrame:
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean()
    return df

def calculate_vwap(df:pd.DataFrame, period: int) -> pd.DataFrame:
    df['price'] = ((df['high'] + df['open'] + df['close'])/3)
    df['price_x_volume'] = df['price'] * df['volume']
    df['cumulative_price_x_volume'] = df.groupby(df.index.date)['price_x_volume'].cumsum()
    df['cumulative_volume'] = df.groupby(df.index.date)['volume'].cumsum()
    df['vwap'] = df['cumulative_price_x_volume'] / df['cumulative_volume']
    return df
    
def calculate_vwap_zscore(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the VWAP Z-score for a given DataFrame.
    Assumes 'price' and 'vwap' columns exist.
    """
    # Calculate the difference between price and VWAP
    df['price_minus_vwap'] = df['price'] - df['vwap']

    # Calculate the rolling standard deviation of the price_minus_vwap
    # This standard deviation represents the volatility around the VWAP
    df['std_dev_price_minus_vwap'] = df.groupby(df.index.date)['price_minus_vwap'].rolling(window=period).std().reset_index(level=0, drop=True)

    # Calculate the Z-score
    df['vwap_zscore'] = df['price_minus_vwap'] / df['std_dev_price_minus_vwap']
    return df


def calculate_mfi(df: pd.DataFrame, period: int) -> pd.DataFrame:

    # Calculate Typical Price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate Raw Money Flow
    rmf = tp * df['volume']
    
    # Create money flow direction
    mf_direction = tp.diff()
    positive_mf = rmf.where(mf_direction > 0, 0)
    negative_mf = rmf.where(mf_direction < 0, 0)
    
    # Calculate money flow ratio
    mfr = (
        positive_mf.rolling(period).sum() / 
        negative_mf.rolling(period).sum()
    ).replace([np.inf, -np.inf], np.nan)
    
    # Calculate MFI
    df['MFI'] = 100 - (100 / (1 + mfr))
    return df

def calculate_macd(df):
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    return df


def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Calculate all required technical indicators."""
    df = calculate_bollinger_bands(df, Config.BB_WINDOW, Config.BB_STD)
    df = calculate_rsi(df, Config.RSI_PERIOD)
    df = calculate_stochastic(df, Config.STOCH_PERIOD)
    df = calculate_accumulation_distribution(df, Config.AD_LOOKBACK)
    df = calculate_atr(df, Config.ATR_PERIOD)
    df = calculate_mfi(df, Config.MFI_Period)
    df = calculate_sma(df, Config.SMA_PERIOD)
    df = calculate_EMA(df)
    df = calculate_vwap(df, Config.vwap_period)
    df = calculate_vwap_zscore(df, 20)
    df = calculate_macd(df)
    df['Volume_MA'] = df['volume'].rolling(window=50).mean()
    df['Bullish_Divergence'] = calculate_bullish_divergence(df, lookback=25)
    
    # Low pierced the band, but close finished inside it
    df['BB_Reversal'] = (df['low'] < df['BB_DN']) & (df['close'] > df['BB_DN'])

    # Drop rows where any of the required indicators are NaN
    required = ['RSI', '%K', '%K_slope', 'AD', 'AD_LOW', 'ATR', 'MFI', 'SMA', '%K_slope_3','EMA_20','EMA_50','EMA_100','Bullish_Divergence','Volume_MA','%D','vwap','macd','macd_signal','macd_line'
                ]
    return df.ffill().dropna(subset=['RSI','%K','ATR'])


# ----------------------------------------------------------------------
# POSITION SIZING
# ----------------------------------------------------------------------
def calculate_position_size(
    current_equity: float, 
    atr: float, 
    entry_price: float,
    leverage: float = Config.LEVERAGE
) -> Dict[str, float]:
    """
    Calculate position size based on ATR and risk management rules.
    Returns dictionary with position details.
    """
    # Maximum risk per trade in dollars
    max_risk_dollars = Config.INITIAL_EQUITY * Config.MAX_RISK_PER_TRADE
    
    # Calculate stop distance based on our stop loss percentage
    stop_distance = entry_price * Config.STOP_LOSS_PCT
    
    # Base position size (before leverage)
    base_position_value = max_risk_dollars 
    
    # Apply leverage
    leveraged_position_value = base_position_value * leverage
    
    # Apply maximum position size constraint
    max_position_value = current_equity * Config.MAX_POSITION_PCT * leverage
    final_position_value = min(leveraged_position_value, max_position_value)
    
    # Calculate number of shares
    shares = int(max_risk_dollars / entry_price) * 2
    
    # Actual position value
    actual_position_value = shares * entry_price
    
    return {
        'shares': shares,
        'position_value': actual_position_value,
        'risk_amount': actual_position_value * Config.STOP_LOSS_PCT / leverage,
        'position_pct': actual_position_value / (current_equity * leverage)
    }


# ----------------------------------------------------------------------
# SIGNAL GENERATION
# ----------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Generate trading signals with filters added."""
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Lower_Wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['Body_Size'] = (df['close'] - df['open']).abs()

    lookback_down = 30  # minutes

    #Sample calculation for entry signal
    recently_oversold = (
        (df['vwap_zscore'].rolling(lookback_down).min() < -1.5) |
        (df['RSI'].rolling(lookback_down).min() < 30)
    )
    df['Recently_Oversold'] = recently_oversold
    lookback_high = 10  # bars

    prior_swing_high = df['high'].rolling(lookback_high).max().shift(1)
    df['Break_of_Structure'] = df['close'] > prior_swing_high

   #Long signal - Strategy logic redacted
    df['LongSignal'] = (
        #FILLER LOGIC
        df['Break_of_Structure'] & 
        df['TimeOK'] 

    )

    return df

# ----------------------------------------------------------------------
# BACKTESTING ENGINE
# ----------------------------------------------------------------------
def should_liquidate(current_time: pd.Timestamp) -> bool:
    """Check if current time is at or after liquidation time"""
    liquidation_time = time(15, 30)  # 3:30 PM
    return current_time.time() >= liquidation_time


def find_exit(
    df: pd.DataFrame,
    start_idx: int,
    entry_date: pd.Timestamp,
    entry_price: float,
    entry_time: pd.Timestamp,
    stop_price: float,
    target_price: float,
) -> Dict:
    """
    Fortified exit logic that correctly handles price gaps for stop-loss
    and take-profit orders.
    """
    n = len(df)
    j = start_idx + 1
    entered_overbought_zone = False
    
    # Optional: Breakeven logic could be implemented here

    while j < n:
        # --- Get current bar's data ---
        cur_timestamp = df.index[j]
        cur_date = cur_timestamp.date()
        open_price = df['open'].iat[j]
        low = df['low'].iat[j]
        high = df['high'].iat[j]
        close = df['close'].iat[j]
        cur_k = df['%K'].iat[j]
        cur_rsi  = df['%K'].iat[j]

        # --- Priority 1: End of Day / Session Liquidation ---
        if should_liquidate(cur_timestamp):
            return {'price': close, 'type': '3:30 PM Liquidation', 'idx': j}
        if cur_date != entry_date:
            return {'price': df['open'].iat[j], 'type': 'Overnight Gap Exit', 'idx': j} #SHOULD NEVER TRIGGER

        # --- Priority 3: INTRADAY Stop-Loss or Take-Profit ---
        #did price make it fully through stop loss or TP - extra safety net for slippage
        if low <= stop_price:
            return {'price': stop_price, 'type': 'Stop Loss', 'idx': j}
        if high >= target_price:
            return {'price': target_price, 'type': 'Take Profit', 'idx': j}
        
        entered_sell_zone = True
        # --- Priority 4: Discretionary/Dynamic Exits ---
        if entered_sell_zone :
            return {'price': close, 'type': 'Momentum/Dynamic Exit', 'idx': j}


        j += 1

    # Final fallback exit at the very end of the data
    return {'price': df['close'].iat[n - 1], 'type': 'End of Data', 'idx': n - 1}


def backtest(df: pd.DataFrame, config: Config, ticker:str) -> pd.DataFrame:
    """
    Backtest using LIMIT ORDERS at the Signal Candle's Close.
    Validates if the next candle's Low is low enough to fill the order.
    """
    trades = []
    current_equity = config.INITIAL_EQUITY
    i = 0
    n = len(df)
    
    # Metrics for Limit Order Analysis
    missed_trades = 0
    current_trade_date = df.index[0].date()
    day_start_equity = current_equity
    paused_for_day = False
    
    while i < n - 1:
        current_timestamp = df.index[i]
        current_date = current_timestamp.date()

        if current_date != current_trade_date:
            # 1. Reset the baseline equity for the new day
            day_start_equity = current_equity
            # 2. Reset the pause flag
            paused_for_day = False
            # 3. Update tracker
            current_trade_date = current_date
            
            # (Optional Debug)
            # print(f"--- New Day: {current_date} | Start Eq: ${day_start_equity:.2f} ---")

        # ---  PAUSE CHECK ---
        # If hit the circuit breaker, stop trading
        if paused_for_day:
            i += 1
            continue

        # 1. Liquidation Check
        if should_liquidate(df.index[i]):
            i += 1
            continue

        # 2. Signal Check
        if df['LongSignal'].iat[i]:
            # --- LIMIT ORDER LOGIC ---
            # Strategy: Place Limit Buy at the Close of the Signal Bar.
            limit_price = df['close'].iat[i]
            
            # Inspect the NEXT candle to see if we get filled
            next_open = df['open'].iat[i+1]
            next_low = df['low'].iat[i+1]
            next_timestamp = df.index[i+1]
            
            fill_price = 0.0
            filled = False
            
            # Condition A: Gap Down (Open < Limit) -> Fill at Open (Better Price)
            if next_open <= limit_price:
                fill_price = next_open
                filled = True
                
            # Condition B: Wick Down (Low <= Limit) -> Fill at Limit
            elif next_low <= limit_price:
                fill_price = limit_price
                filled = True
                
            # Condition C: Price Runs Away (Low > Limit) -> No Fill
            else:
                filled = False
                missed_trades += 1
                # We missed the boat. Move to next bar to look for new signal.
                i += 1
                continue

            # --- EXECUTION (If Filled) ---
            entry_date = next_timestamp.date()
            atr = df['ATR'].iat[i]
            rsi = df['RSI'].iat[i]
            stochastic = df['%K'].iat[i]
            ad = df['AD_LOW'].iat[i]
            
            # Position Sizing
            position_info = calculate_position_size(
                current_equity=current_equity,
                atr=atr,
                entry_price=fill_price,
                leverage=config.LEVERAGE
            )
            
            if position_info['shares'] < 1:
                i += 1
                continue
            
            # Stops/Targets based on FILL PRICE
            stop_price = fill_price * (1 - config.STOP_LOSS_PCT)
            target_price = fill_price * (1 + config.TAKE_PROFIT_PCT)
            
            exit_info = find_exit(
            df, i + 1, entry_date, fill_price, next_timestamp, 
            stop_price, target_price
            )
            
            # --- EXIT SLIPPAGE ONLY ---
            # paid 0 slippage on entry (Limit Order).
            # pay slippage on exit (Market Order for Stop/TP).
            exit_price_ideal = exit_info['price']
            exit_atr = df['ATR'].iat[exit_info['idx']]
            
            exit_slippage = 0
            if config.SLIPPAGE_ENABLED:
                exit_slippage = exit_atr * config.SLIPPAGE_ATR_MULTIPLIER
                
            # For Long Exit (Sell), slippage reduces price
            exit_price_actual = exit_price_ideal - exit_slippage
            
            # PnL Calculation
            pnl_gross = (exit_price_actual - fill_price) * position_info['shares']
            
            commission = 0
            if config.COMMISSIONS_ENABLED:
                commission = max(config.COMMISSION_PER_SHARE * position_info['shares'], config.MIN_COMMISSION_PER_ORDER) * 2
                
            pnl_net = pnl_gross - commission
            pnl_pct = (pnl_net / position_info['position_value']) * 100
            
            current_equity += pnl_net
            daily_drawdown = (current_equity - day_start_equity) / day_start_equity
            
            if daily_drawdown <= -config.MAX_DAILY_DRAWDOWN_PCT:
                print('TRADING PAUSED')
                paused_for_day = True

            
            trades.append({
                'TradeId': str(uuid.uuid4()),
                'Ticker': ticker,            
                'Position': 'LONG',
                'SignalDate': df.index[i],
                'EntryDate': next_timestamp,
                'EntryPrice': fill_price,
                'EntryRSI':rsi,
                'Entry %K': stochastic,
                'Entry_accum/dist':ad,
                'ExitDate': df.index[exit_info['idx']],
                'ExitPrice': exit_price_actual,
                'ExitType': exit_info['type'],
                'Shares': position_info['shares'],
                'PositionValue': position_info['position_value'],
                'PnL_%': pnl_pct,
                'PnL_$': pnl_net,
                'Equity': current_equity,
                'Slippage_$': exit_slippage * position_info['shares']
            })
            
            # Fast forward loop to exit index
            i = exit_info['idx'] + 1
        else:
            i += 1
            
    trade_df = pd.DataFrame(trades)
    
    # Debugging Output to analyze Limit Order efficacy
    # This tells us if the strategy is chasing or not
    total_signals = len(trade_df) + missed_trades
    if total_signals > 0:
        fill_rate = len(trade_df) / total_signals
        print(f"  [Limit Order Logic] Signals: {total_signals} | Filled: {len(trade_df)} | Missed: {missed_trades} | Fill Rate: {fill_rate:.1%}")

    return trade_df


# ----------------------------------------------------------------------
# PERFORMANCE METRICS
# ----------------------------------------------------------------------
def calculate_metrics(
    trade_df: pd.DataFrame,
    df: pd.DataFrame,
    config: Config
) -> Dict:
    """Return a dictionary of high‑level performance statistics."""
    if trade_df.empty:
        return {}

    total = len(trade_df)
    wins = (trade_df['PnL_%'] > 0).sum()
    win_rate = wins / total

    # Max drawdown
    equity = trade_df['Equity']
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    max_dd = drawdown.max() * 100

    # Exit type analysis
    exit_types = trade_df['ExitType'].value_counts()
    
    # Average position size
    avg_position_size = trade_df['PositionValue'].mean()

    #Sharpe calculations
    annualized_sharpe = 0
    if not trade_df.empty:
        #  Resample trades to get the PnL for each calendar day
        # Calculated based on equity
        trade_df['Date'] = pd.to_datetime(trade_df['ExitDate']).dt.date
        daily_equity = trade_df.groupby('Date')['Equity'].last()
        daily_returns_pct = daily_equity.pct_change().fillna(0)

        # Create a full series of all trading days from the original price data
        all_trading_days = df.index.to_series().dt.date.unique()

        # Reindex the daily returns to include days with zero trades (essential!)
        full_daily_returns = daily_returns_pct.reindex(all_trading_days, fill_value=0)

        # Calculate the annualized Sharpe Ratio
        if full_daily_returns.std() > 0:
            # Assumes a risk-free rate of 0 for simplicity
            daily_sharpe = full_daily_returns.mean() / full_daily_returns.std()
            annualized_sharpe = daily_sharpe * np.sqrt(252) # Standard 252 trading days

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'avg_pnl_pct': trade_df['PnL_%'].mean(),
        'best_trade_pct': trade_df['PnL_%'].max(),
        'worst_trade_pct': trade_df['PnL_%'].min(),
        'total_pnl_dollars': trade_df['PnL_$'].sum(),
        'final_equity': trade_df['Equity'].iloc[-1] if not trade_df.empty else config.INITIAL_EQUITY,
        'max_drawdown_pct': max_dd,
        
        # --- MODIFIED: Renamed old Sharpe and added the new one ---
        'sharpe_ratio_per_trade': (trade_df['PnL_%'].mean() / trade_df['PnL_%'].std())
        if trade_df['PnL_%'].std() > 0
        else 0,
        'sharpe_ratio_annualized': annualized_sharpe,
        
        'exit_types': exit_types.to_dict(),
        'avg_position_size': avg_position_size,
    }
def analyze_weekly_performance(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups trades by week and calculates statistics for each week.
    Helps identify if performance is consistent or lumpy.
    """
    if trade_df.empty:
        return pd.DataFrame()

    # Ensure ExitDate is datetime
    df = trade_df.copy()
    df['ExitDate'] = pd.to_datetime(df['ExitDate'])
    
    # Set index to ExitDate for resampling
    df.set_index('ExitDate', inplace=True)

    # Resample by Week ('W-FRI' means weeks ending on Friday)
    weekly_stats = df.resample('W-FRI').agg({
        'PnL_$': 'sum',           # Total PnL for the week
        'PnL_%': 'sum',           # Total Return % for the week
        'Shares': 'count',        # Number of trades (using Shares col as proxy)
        'Equity': 'last'          # Equity at end of week
    })

    # Rename 'Shares' to 'TradeCount' for clarity
    weekly_stats.rename(columns={'Shares': 'TradeCount'}, inplace=True)

    # Calculate Win Rate per week
    # We have to do this separately because standard agg functions are limited
    weekly_wins = df[df['PnL_$'] > 0].resample('W-FRI')['PnL_$'].count()
    weekly_stats['Wins'] = weekly_wins
    weekly_stats['WinRate'] = (weekly_stats['Wins'] / weekly_stats['TradeCount']).fillna(0)

    # Clean data
    weekly_stats = weekly_stats.drop(columns=['Wins'])

    return weekly_stats

def plot_daily_pnl(trade_df: pd.DataFrame, ticker: str, output_dir: Path):
    """
    Generates a dual-axis chart with readable dates.
    Auto-adjusts x-axis intervals based on backtest duration.
    """
    if trade_df.empty:
        print("⚠️ Cannot plot PnL: No trades found.")
        return

    # Prepare Daily Data
    df = trade_df.copy()
    # Ensure ExitDate is a datetime object first
    df['ExitDate'] = pd.to_datetime(df['ExitDate']) 
    df['Date'] = df['ExitDate'].dt.date
    
    daily_stats = df.groupby('Date').agg({
        'PnL_$': 'sum',
        'Equity': 'last'
    }).sort_index()
    
    # Fill gaps for smoother line chart
    idx = pd.date_range(daily_stats.index.min(), daily_stats.index.max())
    daily_stats = daily_stats.reindex(idx).fillna(0)
    
    # Forward fill Equity (it stays flat on non-trading days)
    if daily_stats['Equity'].iloc[0] == 0:
         daily_stats['Equity'].iloc[0] = Config.INITIAL_EQUITY
    daily_stats['Equity'] = daily_stats['Equity'].replace(to_replace=0, method='ffill')

    #  Setup Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar Chart (Daily PnL)
    colors = ['green' if x >= 0 else 'red' for x in daily_stats['PnL_$']]
    ax1.bar(daily_stats.index, daily_stats['PnL_$'], color=colors, alpha=0.6, label='Daily PnL', width=0.8)
    ax1.set_ylabel('Daily PnL ($)', fontsize=12)
    
    # Line Chart (Equity)
    ax2 = ax1.twinx()
    ax2.plot(daily_stats.index, daily_stats['Equity'], color='blue', linewidth=2, label='Account Equity')
    ax2.set_ylabel('Total Equity ($)', color='blue', fontsize=12)
    
    # Calculate how many days total are in the chart
    total_days = (daily_stats.index.max() - daily_stats.index.min()).days
    
    # Set locator based on duration to prevent crowding
    if total_days <= 30:
        locator = mdates.DayLocator(interval=2)
        fmt = mdates.DateFormatter('%m-%d')
    elif total_days <= 90:
        locator = mdates.WeekdayLocator(interval=1)
        fmt = mdates.DateFormatter('%b %d')
    elif total_days <= 365:
        locator = mdates.MonthLocator(interval=1)
        fmt = mdates.DateFormatter('%b %Y')
    else:
        locator = mdates.MonthLocator(interval=3)
        fmt = mdates.DateFormatter('%b %Y')

    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(fmt)
    
    # Auto-format dates (rotates them 45 degrees and aligns them right)
    fig.autofmt_xdate()

    # Title and Legends
    plt.title(f"{ticker} - PnL & Equity ({total_days} Days)", fontsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{ticker}_PnL_Graph_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"📈 PnL Graph saved to: {filename}")

def apply_portfolio_logic(all_trades_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Simulates a live portfolio by replaying all potential trades chronologically.
    Enforces a Global Daily Circuit Breaker.
    """
    print("\n STARTING PORTFOLIO REPLAY SIMULATION...")
    
    # Create Event Timeline
    # We split trades into two events: ENTRY and EXIT
    entries = all_trades_df[['EntryDate', 'TradeId']].copy()
    entries.columns = ['Timestamp', 'TradeId']
    entries['Type'] = 'ENTRY'
    
    exits = all_trades_df[['ExitDate', 'TradeId', 'PnL_$']].copy()
    exits.columns = ['Timestamp', 'TradeId', 'PnL_$']
    exits['Type'] = 'EXIT'
    
    # Merge and Sort by Time
    timeline = pd.concat([entries, exits]).sort_values(by=['Timestamp', 'Type'])
    
    # Replay Variables
    current_equity = config.INITIAL_EQUITY
    day_start_equity = current_equity
    current_date = timeline['Timestamp'].iloc[0].date()
    
    paused_for_day = False
    active_trade_ids = set() # Trades that were allowed to enter
    accepted_trades = []
    
    # Iterate Time
    for index, event in timeline.iterrows():
        event_date = event['Timestamp'].date()
        
        if event_date != current_date:
            day_start_equity = current_equity
            paused_for_day = False
            current_date = event_date
            
        if event['Type'] == 'ENTRY':
            if paused_for_day:
                continue 

            active_trade_ids.add(event['TradeId'])
            
        elif event['Type'] == 'EXIT':
            if event['TradeId'] in active_trade_ids:
                # Update Equity
                pnl = event['PnL_$']
                current_equity += pnl
                
                daily_pnl = current_equity - day_start_equity
                drawdown_pct = daily_pnl / day_start_equity
                
                if not paused_for_day and drawdown_pct <= -config.MAX_DAILY_DRAWDOWN_PCT:
                    paused_for_day = True
                    
                # Store the valid trade ID to reconstruct DataFrame later
                accepted_trades.append(event['TradeId'])

    # Filter the original DataFrame
    # Keep only trades that survived the replay
    final_df = all_trades_df[all_trades_df['TradeId'].isin(accepted_trades)].copy()
    
    # Recalculate Equity Curve for the final DataFrame
    final_df = final_df.sort_values('ExitDate')
    final_df['Equity'] = config.INITIAL_EQUITY + final_df['PnL_$'].cumsum()
    
    return final_df

def generate_report(ticker: str, trade_df: pd.DataFrame, metrics: Dict, output_dir: Path) -> None:
    """Creates a text report and summary CSV for quick review."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{ticker}_results_{timestamp}.csv"
    txt_filename = f"{ticker}_summary_{timestamp}.txt"

    trade_df.to_csv(output_dir / csv_filename, index=False)

    with open(output_dir / txt_filename, "w") as f:
        f.write(f"BACKTEST REPORT: {ticker}\n")
        f.write(f"Period: {Config.START} → {Config.END}\n")
        f.write(f"Initial Equity: ${Config.INITIAL_EQUITY}\n")
        f.write(f"Final Equity: ${metrics['final_equity']:.2f}\n")
        f.write(f"Total PnL: ${metrics['total_pnl_dollars']:.2f}\n")
        f.write(f"Win Rate: {metrics['win_rate']*100:.1f}%\n")
        f.write(f"Sharpe: {metrics['sharpe_ratio_annualized']:.2f}\n")
        f.write(f"Max DD: {metrics['max_drawdown_pct']:.2f}%\n")
        f.write("\nExit Types:\n")
        for k, v in metrics['exit_types'].items():
            f.write(f"  {k:<20} {v:>5}\n")

# MAIN EXECUTION
# ----------------------------------------------------------------------
def main() -> None:
    """Run the whole pipeline for each ticker defined in Config."""
    # ------------------------------------------------------------------
    # 1) Initialize Alpaca API
    # ------------------------------------------------------------------
    api_key = 'API KEY HERE'
    api_secret ='SECRET API  KEY HERE'
    base_url = 'https://paper-api.alpaca.markets/v2'

    api = tradeapi.REST(api_key, api_secret, base_url)
    all_potential_trades = []
    portfolio_results= []

    # ------------------------------------------------------------------
    # 3) Loop over tickers
    # ------------------------------------------------------------------
    for ticker in Config.TICKERS:
        print(f"\n{'='*60}")
        print(f"BACKTESTING {ticker} | {Config.START} → {Config.END}")
        print(f"Leverage: {Config.LEVERAGE}x | Liquidation Time: {Config.LIQUIDATION_TIME}")
        print(f"{'='*60}")

        # ---- Download data -------------------------------------------------
        print(f"\n Downloading {ticker} data...")
        df = download_intraday_chunks(
            api,
            ticker,
            Config.START,
            Config.END,
            Config.INTERVAL,
            Config.CHUNK_DAYS,
        )

        if df.empty:
            print(" No market data retrieved.")
            continue


        # ---- Calculate indicators ------------------------------------------
        print(" Calculating technical indicators...")
        df = calculate_indicators(df, Config)

        # ---- Generate signals ----------------------------------------------
        print(" Generating trading signals...")
        df = generate_signals(df, Config)
        
        # ---- Run backtest --------------------------------------------------
        print("\n Running backtest...")
        trade_df = backtest(df, Config, ticker)
    
        if not trade_df.empty:
            all_potential_trades.append(trade_df)

        if trade_df.empty:
            print("  No trades generated for this period.")
            continue
    

        # ---- Compute and display metrics -----------------------------------
        metrics = calculate_metrics(trade_df, df, Config)
        
        print(f"\n PERFORMANCE SUMMARY")
        print(f"{'─'*40}")
        print(f"Total trades      : {metrics['total_trades']}")
        print(f"Win rate          : {metrics['win_rate']:.1%}")
        print(f"Avg P&L per trade : {metrics['avg_pnl_pct']:.2f}%")
        print(f"Best trade        : {metrics['best_trade_pct']:.2f}%")
        print(f"Worst trade       : {metrics['worst_trade_pct']:.2f}%")
        print(f"Total P&L         : ${metrics['total_pnl_dollars']:.2f}")
        print(f"Final equity      : ${metrics['final_equity']:.2f}")
        print(f"Total return      : {((metrics['final_equity']/Config.INITIAL_EQUITY)-1)*100:.2f}%")
        print(f"Max drawdown      : {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe ratio      : {metrics['sharpe_ratio_annualized']:.2f}")
        
        print(f"\n POSITION SIZING")
        print(f"{'─'*40}")
        print(f"Avg position size : ${metrics['avg_position_size']:.2f}")
        
        print(f"\n EXIT ANALYSIS")
        print(f"{'─'*40}")
        for exit_type, count in metrics['exit_types'].items():
            print(f"{exit_type:<20}: {count:3d} ({count/metrics['total_trades']*100:5.1f}%)")

        portfolio_results.append({
            'Ticker': ticker,
            'Trades': metrics['total_trades'],
            'Win_Rate': metrics['win_rate'],
            'Sharpe': metrics['sharpe_ratio_annualized'],
            'PnL': metrics['total_pnl_dollars'],
            'Max_DD': metrics['max_drawdown_pct'],
            'Avg_Return': metrics['avg_pnl_pct']
        })

        
        weekly_df = analyze_weekly_performance(trade_df)
        
        if not weekly_df.empty:
            # Format the output nicely
     
            
            for date, row in weekly_df.iterrows():
                # Only print weeks that had trades (optional)
                if row['TradeCount'] > 0:
                    date_str = date.strftime('%Y-%m-%d')
                   
            
            # Save weekly stats to CSV separately if you want
            weekly_filename = f"backtest_results/{ticker}_weekly_stats.csv"
            weekly_df.to_csv(weekly_filename)
            print(f"\n Weekly stats saved to: {weekly_filename}")
        else:
            print("No weekly data available.")

        if not all_potential_trades:
            print("No trades generated across portfolio.")
            return

        master_df = pd.concat(all_potential_trades)
        print(f"\n Total Potential Trades Generated: {len(master_df)}")

        # 3. REPLAY PHASE (Apply Portfolio Logic)
        final_portfolio_df = apply_portfolio_logic(master_df, Config)
        
        print(f" Final Executed Trades: {len(final_portfolio_df)}")
        print(f" Trades Blocked by Circuit Breaker: {len(master_df) - len(final_portfolio_df)}")

        # 4. REPORTING PHASE (Global Stats)
        metrics = calculate_metrics(final_portfolio_df, df, Config) # Note: 'df' here is just for dates, might need adjustment
        
        print(f"\n PORTFOLIO PERFORMANCE")
        print(f"{'='*40}")
        print(f"Final Equity: ${metrics['final_equity']:.2f}")
        print(f"Total PnL:    ${metrics['total_pnl_dollars']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Generate the Portfolio Graph
        output_dir = Path("backtest_results")
        plot_daily_pnl(final_portfolio_df, "PORTFOLIO_COMBINED", output_dir)

        # ---- Time analysis -------------------------------------------------

        # ---- Sample trades -------------------------------------------------
        print(f"\n SAMPLE TRADES (First 10)")
        print(f"{'─'*40}")
        sample_columns = [
            'SignalDate', 'EntryPrice', 'ExitDate', 'ExitPrice', 
            'ExitType', 'Shares', 'PnL_%', 'Equity'
        ]
        print(trade_df.head(10)[sample_columns].to_string(index=False))
        
        # ---- Additional Analysis -------------------------------------------
        print(f"\n TRADE DURATION ANALYSIS")
        print(f"{'─'*40}")
        trade_df['Duration'] = pd.to_datetime(trade_df['ExitDate']) - pd.to_datetime(trade_df['SignalDate'])
        trade_df['DurationMinutes'] = trade_df['Duration'].dt.total_seconds() / 60
        
        avg_duration = trade_df['DurationMinutes'].mean()
        print(f"Average trade duration: {avg_duration:.1f} minutes")
        
        # Duration by exit type
        duration_by_exit = trade_df.groupby('ExitType')['DurationMinutes'].agg(['mean', 'count'])
        print("\nDuration by Exit Type:")
        print(duration_by_exit.round(1))

        if not portfolio_results:
            print("\n No results to aggregate.")
            return

        # Convert to DataFrame for analysis
        summary_df = pd.DataFrame(portfolio_results)

        # A. Overall Statistics
        avg_sharpe = summary_df['Sharpe'].mean()
        median_sharpe = summary_df['Sharpe'].median()
        total_pnl = summary_df['PnL'].sum()
        avg_pnl = summary_df['PnL'].mean()
        profitable_tickers = summary_df[summary_df['PnL'] > 0].shape[0]
        total_tickers = len(summary_df)
        win_pct = (profitable_tickers / total_tickers) * 100

        # B. Top 10 Performers
        top_10 = summary_df.sort_values(by='Sharpe', ascending=False).head(10)
        
        # C. Print Report
        print(f"\n\n{'#'*60}")
        print(f"FINAL BACKTEST SUMMARY (CONDSET_D_FINAL)")
        print(f"{'#'*60}")
        print(f"Tickers Tested    : {total_tickers}")
        print(f"Profitable Tickers: {profitable_tickers} ({win_pct:.1f}%)")
        print(f"Total Portfolio PnL: ${total_pnl:,.2f}")
        print(f"Average PnL/Ticker : ${avg_pnl:,.2f}")
        print(f"{'-'*30}")
        print(f"AVERAGE SHARPE     : {avg_sharpe:.4f}")
        print(f"MEDIAN SHARPE      : {median_sharpe:.4f}")
        print(f"{'-'*30}")
        
        print(f"\n TOP 10 PERFORMERS (By Sharpe)")
        print(top_10.to_string(index=False, float_format="%.4f"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_df.to_csv(f"backtest_results/FULL_SUMMARY_{timestamp}_CONDSET_D_FINAL.csv", index=False)
        print(f"\n Full summary saved to backtest_results/FULL_SUMMARY_{timestamp}_CONDSET_D_FINAL.csv")
        
        # ---- Export results ------------------------------------------------
        current_dir = os.getcwd()
        
        # Create output directory
        output_dir = Path(current_dir) / "backtest_results"
        output_dir.mkdir(exist_ok=True)

        try:
            plot_daily_pnl(trade_df, ticker, output_dir)
        except Exception as e:
            print(f" Error generating graph: {e}")

        
        # Create filename with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{ticker}_backtest_{Config.START}_{Config.END}_{timestamp}.csv"
        csv_filepath = output_dir / csv_filename
        
        # Save CSV file
        try:
            trade_df.to_csv(csv_filepath, index=False)
            # Verify file was created
            if csv_filepath.exists():
                file_size = csv_filepath.stat().st_size
                print(f"\n CSV file saved successfully!")
            else:
                print(f"\n Error: File was not created at {csv_filepath}")
        except Exception as e:
            print(f"\n Error saving CSV file: {e}")
        
        # Also save a summary text file
        summary_filename = f"{ticker}_summary_{Config.START}_{Config.END}_{timestamp}.txt"
        summary_filepath = output_dir / summary_filename
        
        try:
            with open(summary_filepath, 'w') as f:
                f.write(f"Backtest Summary for {ticker}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Period: {Config.START} to {Config.END}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"Total trades: {metrics['total_trades']}\n")
                f.write(f"Win rate: {metrics['win_rate']:.1%}\n")
                f.write(f"Average P&L per trade: {metrics['avg_pnl_pct']:.2f}%\n")
                f.write(f"Best trade: {metrics['best_trade_pct']:.2f}%\n")
                f.write(f"Worst trade: {metrics['worst_trade_pct']:.2f}%\n")
                f.write(f"Final equity: ${metrics['final_equity']:.2f}\n")
                f.write(f"Total return: {((metrics['final_equity']/Config.INITIAL_EQUITY)-1)*100:.2f}%\n")
                f.write(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
                f.write(f"Sharpe ratio: {metrics['sharpe_ratio_annualized']:.2f}\n\n")
                
                f.write(f"Exit Type Breakdown:\n")
                for exit_type, count in metrics['exit_types'].items():
                    f.write(f"  {exit_type}: {count} ({count/metrics['total_trades']*100:.1f}%)\n")
            
            if summary_filepath.exists():
                print(f"\n Summary file saved successfully!")
        except Exception as e:
            print(f"\n Error saving summary file: {e}")
        


if __name__ == "__main__":
    main()