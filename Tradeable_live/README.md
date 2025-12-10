# Live Execution Engine

This repository contains the live trading engine designed for implementation on Alpaca Markets.

*Note: While the execution architecture, state management, and latency optimizations are fully available, the alpha logic has been redacted.*

Key Technical Capabilities

 1. Architecture & Latency
- Event-Driven: Uses full-duplex WebSockets via `alpaca-py` for real-time market data.
- Asynchronous I/O: Built on Python's `asyncio` loop to handle non-blocking order management while simultaneously processing data streams.
- O(1) Optimization: Implements a fixed-size rolling window for indicator calculation. Updates are performed via vectorized operations on the tail.

 2. Execution & Resilience
Unlike simple scripts, this engine includes fault-tolerance patterns for 24/7 uptime:
- State Reconciliation: Background tasks run every 60s to reconcile local state with the broker, automatically correcting position drift.
- Background Polling: Fetches account equity in a background thread without stalling the trade loop.
- Atomic Order Management: Uses OCO (One-Cancels-Other) Bracket Orders to submit Entry, Stop-Loss, and Take-Profit in a single API call.

 3. Production Features
- Optimistic State Updates: Updates local state immediately upon execution while awaiting API confirmation.
- Dynamic Sizing: Real-time volatility adjusted sizing based on live account equity.
- Automated Liquidation: Hard coded EOD liquidation logic to prevent overnight holding risk.

 Technology Stack
- Core: Python , Asyncio
-*Data: Pandas, NumPy 
-*API: Alpaca-py SDK 
- Utils: Nest_Asyncio, Pytz

 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Enter Alpaca Keys in main
3. Run program: 
python TradeableLive.py