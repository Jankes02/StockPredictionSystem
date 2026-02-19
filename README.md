# Stock Prediction System

Multi-agent stock trading system for the Polish market (GPW), focused on WIG20. Signal agents (MACD, RSI, ROC, Bollinger, MA trend) produce buy/sell/hold signals with confidence; a decision agent aggregates them into final trades. Built for daily backtesting and for use in a master's thesis (computer science).

## Requirements

- Python 3.x
- Dependencies: `pip install -r requirements.txt`  
  (pandas, numpy, matplotlib, PyYAML)

## Project layout

- **Root:** `main.py` (entry point), `config.py` (loads `config.yaml`), `config.yaml`, `requirements.txt`
- **src/utils:** backtesting, portfolio simulator, metrics, plotting
- **src/agents:** signal agents (MACD, RSI, ROC, Bollinger, MATrend) and the decision agent
- **src/agents/base:** `BaseAgent`, `AgentSignal`, `Decision`
- **data/daily:** CSV price data (one file per symbol, e.g. `ale.csv`)

## Run

From the project root:

```bash
python main.py
```

This loads `config.yaml`, runs a daily backtest on the configured symbols and agents, prints metrics (return, CAGR, max drawdown, Sharpe, Calmar, trades, win rate) and shows an equity + drawdown plot.

## Configuration

Edit **config.yaml** at the project root:

- **data:** `daily_dir` — path to CSV folder (relative to project root)
- **symbols:** list of WIG20 tickers
- **backtest:** `initial_cash`, `data_frequency` (`daily`)
- **portfolio:** `position_size` — fraction of cash per new position
- **decision:** `mode` (`passive` / `balanced` / `aggressive`), `min_confidence`, `max_positions` (or `null` for no limit)
- **agents:** which agents to run and their parameters (e.g. `rsi: { period: 14 }`). Comment out or remove an agent to disable it.

## Data

Place one CSV per symbol in `data/daily/`. Each file should have at least a **Date** and **Close** column; symbol filenames are lowercase (e.g. `ale.csv` for ALE). Symbols listed in `config.yaml` but without a file in the data folder are skipped.

## License

Copyright © 2026 Rafał Jankowski. All rights reserved.

This software is for **academic and research use only**. Commercial use, distribution, sublicensing, or modification for commercial purposes is prohibited without explicit written permission from the author.

See [LICENSE](LICENSE) for full terms.
