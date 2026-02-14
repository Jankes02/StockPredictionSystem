from typing import Dict, List, Optional, Sequence

import pandas as pd

from agents.base.BaseAgent import BaseAgent
from agents.DecisionAgent import DecisionAgent
from PortfolioSimulator import PortfolioSimulator


def backtest_portfolio_daily(
    agents: Sequence[BaseAgent],
    data_by_symbol: Dict[str, pd.DataFrame],
    initial_cash: float = 100_000,
    position_size: float = 0.1,
    decision_agent: Optional[DecisionAgent] = None,
) -> PortfolioSimulator:
    if not data_by_symbol:
        return PortfolioSimulator(initial_cash, position_size)

    decision_agent = decision_agent or DecisionAgent(mode="aggressive")
    portfolio = PortfolioSimulator(initial_cash, position_size)

    symbols = list(data_by_symbol.keys())
    common_dates = data_by_symbol[symbols[0]].index
    for sym in symbols[1:]:
        common_dates = common_dates.intersection(data_by_symbol[sym].index)
    common_dates = common_dates.sort_values()

    for date in common_dates:
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
            symbol: float(data_by_symbol[symbol].loc[date]["Close"])
            for symbol in symbols
        }

        portfolio.process_day(date, prices, decisions)

    return portfolio
