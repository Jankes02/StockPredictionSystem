from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.utils.PortfolioSimulator import PortfolioSimulator


SignalsBySymbolByDate = Dict[pd.Timestamp, Dict[str, List[AgentSignal]]]


def _common_dates(data_by_symbol: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    symbols = list(data_by_symbol.keys())
    common = data_by_symbol[symbols[0]].index
    for sym in symbols[1:]:
        common = common.intersection(data_by_symbol[sym].index)
    return common.sort_values()


def precompute_signals(
    agents: Sequence[BaseAgent],
    data_by_symbol: Dict[str, pd.DataFrame],
) -> SignalsBySymbolByDate:
    """
    Precompute per-date signals for every (agent, symbol) pair using the full
    price history. Because each agent's signal at date t depends only on
    data up to t, this introduces no lookahead leakage when the resulting
    table is later queried over a sub-window.
    """
    signals: SignalsBySymbolByDate = {}
    if not data_by_symbol:
        return signals

    common = _common_dates(data_by_symbol)
    for date in common:
        by_symbol: Dict[str, List[AgentSignal]] = {}
        for symbol, df in data_by_symbol.items():
            data = df.loc[:date]
            if len(data) < 2:
                continue
            by_symbol[symbol] = [agent.generate_signal(data, symbol) for agent in agents]
        signals[date] = by_symbol
    return signals


def backtest_portfolio_daily(
    agents: Sequence[BaseAgent],
    data_by_symbol: Dict[str, pd.DataFrame],
    initial_cash: float = 100_000,
    position_size: float = 0.1,
    decision_agent: Optional[DecisionAgent] = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    stamp_duty_bps: float = 0.0,
    precomputed_signals: Optional[SignalsBySymbolByDate] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    agent_filter: Optional[Sequence[str]] = None,
) -> PortfolioSimulator:
    """
    Run a daily multi-agent backtest.

    If `precomputed_signals` is provided, per-day signals are reused from that
    table instead of re-calling each agent. `start_date` and `end_date` bound
    the trading window (inclusive). `agent_filter`, when given, keeps only
    signals whose `agent` field matches one of the listed names (useful for
    ablation and single-agent baselines).
    """
    simulator_kwargs = {
        "initial_cash": initial_cash,
        "position_size": position_size,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "stamp_duty_bps": stamp_duty_bps,
    }

    if not data_by_symbol:
        return PortfolioSimulator(**simulator_kwargs)

    decision_agent = decision_agent or DecisionAgent(mode="aggressive")
    portfolio = PortfolioSimulator(**simulator_kwargs)

    symbols = list(data_by_symbol.keys())
    dates = _common_dates(data_by_symbol)
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]

    allowed_agents = set(agent_filter) if agent_filter is not None else None

    for date in dates:
        daily_signals: List[AgentSignal] = []

        if precomputed_signals is not None:
            by_symbol = precomputed_signals.get(date, {})
            for symbol in symbols:
                for sig in by_symbol.get(symbol, []):
                    if allowed_agents is None or sig["agent"] in allowed_agents:
                        daily_signals.append(sig)
        else:
            for symbol in symbols:
                data = data_by_symbol[symbol].loc[:date]
                if len(data) < 2:
                    continue
                for agent in agents:
                    if allowed_agents is not None and agent.name not in allowed_agents:
                        continue
                    daily_signals.append(agent.generate_signal(data, symbol))

        decisions = decision_agent.decide(daily_signals)

        prices = {
            symbol: float(data_by_symbol[symbol].loc[date]["Close"])
            for symbol in symbols
            if date in data_by_symbol[symbol].index
        }

        portfolio.process_day(date, prices, decisions)

    return portfolio
