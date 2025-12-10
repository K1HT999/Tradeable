Tradeable Backtester

This repository contains a custom-built backtesting engine designed for intraday strategies on US Equities.

Note: The core execution engine, risk management, and data handling logic are fully available. However, the specific alpha generation entry and exit logic has been redacted and replaced with a standard mean-reversion placeholder for demonstration purposes.*

 Key Technical Capabilities
 1. Infrastructure & Data
- Data Pipeline: Downloads intraday minute-bar OHLCV data from Alpaca and properly converts to correct time zone.
- Data Loading: Downloads data in 7-day chunks to get around Alpaca's Data limits.
- Tweakable parameters: Conditions like slippage, commission, time based stops, etc. are all able to be tweaked in the config class with ease.

 2. The Backtesting Engine 
Unlike off-the-shelf libraries (Backtrader/Zipline), this engine handles real-world microstructure issues:
- Limit Order Simulation: Validates fills based on the *next candle's* Low/Open (not just assuming a fill at close) to best gauge slippage conditions.
- Gap Handling: Logic to handle overnight gaps and intraday liquidity gaps, assuming sub-optimal fills at every signal.
- Dynamic Risk Management: 
- Fixed position size based on risk.
- Daily Circuit Breakers (stops trading if daily drawdown exceeds X%).
- Time-based liquidations.

 3. Feature Engineering
Calculates indicators and metrics using vectorized Pandas operations for speed:
- RSI
- Stochastic values (%K, %R)
- Bullish Divergence
- MACD
- Moving Averages (SMA, EMA)
- ATR
- Accumulation/Distribution
- Bollinger bands
- Money Flow Index 
- Sharpe Ratio
- Max DD
- Week-by-Week analysis
- Trade pausing logic
- Dynamic stop logic

 Technology Stack
- Python Pandas, NumPy, TimeFrame, Dict, itertools
- APIs: Alpaca Trade API
- Visualization: Matplotlib (Automated PnL curve generation)

How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Enter Alpaca Keys in main
3. Run program: 

python TradeableBT.py
