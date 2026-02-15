# Stock Prediction System
<p style='text-align: justify;'>Multi-agent stock trading system for the Polish market (GPW), focused on WIG20. Signal agents (MACD, RSI, ROC, Bollinger, MA trend, etc.) produce buy/sell/hold signals with confidence; a decision agent aggregates them into final trades. Built for backtesting and for use in a master's thesis (computer science).</p>
Multi-agent stock trading system for the Polish market (GPW), focused on WIG20. Signal agents (MACD, RSI, ROC, Bollinger, MA trend, etc.) produce buy/sell/hold signals with confidence; a decision agent aggregates them into final trades. Built for backtesting and for use in a master's thesis (computer science).

## Requirements

- Python 3.x
- Dependencies: `pip install -r requirements.txt`

  (pandas, numpy, matplotlib, PyYAML)

## Project layout

- **Root:** `main.py` (entry point), `config.py` (loads `config.yaml`), `config.yaml`, `requirements.txt`
- **src/utils:** backtesting, portfolio simulator, metrics, plotting
- **src/agents:** signal agents (MACD, RSI, ROC, Bollinger, MATrend, Breakout) and the decision agent
- **data/daily**, **data/5min:** CSV price data (one file per symbol, e.g. `ale.csv`)

## Run

From the project root:

```bash
python main.py
```

This loads `config.yaml`, runs a daily backtest on the configured symbols and agents, prints metrics (return, CAGR, max drawdown, Sharpe, Calmar, trades, win rate) and shows an equity + drawdown plot.

## Configuration

Edit **config.yaml** at the project root:

- **data:** `daily_dir`, `5min_dir` — paths to CSV folders (relative to project root)
- **symbols:** list of WIG20 tickers
- **backtest:** `initial_cash`, `data_frequency` (`daily` or `5min`)
- **portfolio:** `position_size` — fraction of cash per new position
- **decision:** `mode` (`passive` / `balanced` / `aggressive`), `min_confidence`, `max_positions` (or `null` for no limit)
- **agents:** which agents to run and their parameters (e.g. `rsi: { period: 14 }`). Comment out or remove an agent to disable it.

## Data

Place one CSV per symbol in `data/daily/` (and optionally `data/5min/`). Each file should have at least a **Date** and **Close** column; symbol filenames are lowercase (e.g. `ale.csv` for ALE). Symbols listed in `config.yaml` but without a file in the chosen data folder are skipped.

## License

(Add your license here.)
