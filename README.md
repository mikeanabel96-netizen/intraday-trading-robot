# Intraday Trading Robot

A professional intraday trading bot for MetaTrader 5 (MT5) with sniper entries, dynamic support/resistance detection, and intelligent risk management.

## Features

✅ **Sniper Entry Logic** - Wait for confirmed bounces at support/resistance levels
✅ **Dynamic Lot Sizing** - Risk management with 5% risk per trade
✅ **Support & Resistance Detection** - Automatic S/R level identification on M1 timeframe
✅ **Intraday Strategy** - Hold trades for hours, targeting profit on M1 candles
✅ **Backtesting Engine** - Test with CSV historical data
✅ **Order Management** - Automated position tracking and profit taking
✅ **MT5 Integration** - Direct connection to MetaTrader 5

## Project Structure

```
intraday-trading-robot/
├── config/
│   ├── __init__.py
│   ├── settings.py           # Bot configuration
│   └── credentials.py        # MT5 credentials
├── core/
│   ├── __init__.py
│   ├── mt5_connector.py      # MT5 connection handler
│   ├── support_resistance.py # S/R detection
│   ├── lot_sizer.py          # Dynamic lot sizing
│   └── order_manager.py      # Order management
├── strategy/
│   ├── __init__.py
│   ├── sniper_entry.py       # Sniper entry logic
│   └── trading_logic.py      # Main trading strategy
├── backtest/
│   ├── __init__.py
│   ├── backtest_engine.py    # Backtesting system
│   └── data_loader.py        # CSV data loading
├── data/
│   ├── historical/           # CSV files for backtesting
│   └── sample_data.csv       # Sample data
├── logs/
│   └── trading.log           # Trade logs
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mikeanabel96-netizen/intraday-trading-robot.git
cd intraday-trading-robot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your settings in `config/settings.py`

## Configuration

Edit `config/settings.py`:
```python
ACCOUNT_SIZE = 10  # $10 USD
RISK_PER_TRADE = 0.05  # 5%
TIMEFRAME = 1  # M1 candles
TRADING_PAIR = "EURUSD"
MAX_POSITIONS = 5
```

## Usage

### Run Live Trading
```bash
python main.py --mode live
```

### Run Backtesting
```bash
python main.py --mode backtest --data data/historical/sample_data.csv
```

## Strategy Details

### Sniper Entry
- Identifies support and resistance levels on M1
- Waits for confirmed bounce/reversal at these levels
- Enters with tight stop loss
- Targets profit based on risk/reward ratio

### Risk Management
- Dynamic lot sizing: Risk = Account × Risk% / (Entry - StopLoss)
- Maximum 5% risk per trade
- Automatic stop loss placement
- Profit taking at resistance levels

## Requirements

- Python 3.8+
- MetaTrader 5 Terminal
- Historical data (CSV format)

## License

MIT License

## Author

mikeanabel96-netizen

---

**Disclaimer**: This bot is for educational purposes. Use at your own risk. Always backtest thoroughly before live trading.