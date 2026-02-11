from agents.DecisionAgent import DecisionAgent
from PortfolioSimulator import PortfolioSimulator
import numpy as np


def backtest_portfolio_daily(
    agents,
    data_by_symbol: dict,
    initial_cash=100_000,
    confidence=0.05
):
    
    decision_agent = DecisionAgent(mode='aggressive', min_confidence=confidence)
    portfolio = PortfolioSimulator(initial_cash)

    symbols = list(data_by_symbol.keys())
    dates = data_by_symbol[symbols[0]].index

    for date in dates:
        daily_signals = []

        for symbol in symbols:
            data = data_by_symbol[symbol].loc[:date]
            if len(data) < 2:
                continue

            for agent in agents:
                sig = agent.generate_signal(data, symbol)
                daily_signals.append(sig)

        decisions = decision_agent.decide(daily_signals)

        prices = {
            symbol: data_by_symbol[symbol].loc[date]["Close"]
            for symbol in symbols
            if date in data_by_symbol[symbol].index
        }

        portfolio.process_day(date, prices, decisions)

    return portfolio
