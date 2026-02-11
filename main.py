from agents.MACDAgent import MACDAgent
from agents.RSIAgent import RSIAgent
from agents.ROCAgent import ROCAgent
from agents.BollingerAgent import BollingerAgent
from agents.MATrendAgent import MATrendAgent
from utils.backtesting import backtest_portfolio_daily
from MetricsCalculator import MetricsCalculator
from utils.plotting import plot_equity_with_drawdown
import pandas as pd
import numpy as np

daily_data_folder = 'data\\daily'
intraday_data_folder = 'data\\5min'
symbols = [
    'ALE', 'ALR', 'BDX', 'CCC', 'CDR',
    'DNP', 'KGH', 'KRU', 'KTY', 'LPP',
    'MBK', 'OPL', 'PCO', 'PEO', 'PGE',
    'PKN', 'PKO', 'PZU', 'SPL', 'ZAB'
]

data_by_symbol = {}

for symbol in symbols:
    try:
        path = f"{daily_data_folder}\\{symbol.lower()}.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        data_by_symbol[symbol] = df
    except:
        continue
    
if __name__ == "__main__":
    agents = [
        MACDAgent(),
        RSIAgent(),
        ROCAgent(),
        BollingerAgent(),
        MATrendAgent()
    ]

    portfolio = backtest_portfolio_daily(
        agents=agents,
        data_by_symbol=data_by_symbol,
        initial_cash=100_000
    )

    metrics = MetricsCalculator(
        portfolio.equity_curve,
        portfolio.trades
    )
    summary = metrics.summary()
    print(metrics.summary())
    # plot_equity_with_drawdown(portfolio.equity_curve)