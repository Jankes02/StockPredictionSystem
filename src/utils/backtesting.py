from typing import Dict, List, Literal, Optional, Sequence

import pandas as pd

from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.agents.base.Decision import Decision
from src.utils.PortfolioSimulator import PortfolioSimulator


SignalsBySymbolByDate = Dict[pd.Timestamp, Dict[str, List[AgentSignal]]]

# "close": fill on the signal day's Close (the protocol used in the paper; it
#          assumes the order is executed in the closing auction of the same
#          session the signal was computed on).
# "next_open": fill on the next session's Open. This is what a live system can
#          actually do when signals are computed after the close, so it is the
#          default for the live tool.
FillMode = Literal["close", "next_open"]


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

    Each agent is evaluated once per symbol with its vectorised
    `generate_signal_series`, which is tested to be identical to calling
    `generate_signal` on every prefix of the history.
    """
    signals: SignalsBySymbolByDate = {}
    if not data_by_symbol:
        return signals

    common = _common_dates(data_by_symbol)

    # symbol -> list of (agent, frame aligned to common dates, rows-so-far)
    per_symbol = {}
    for symbol, df in data_by_symbol.items():
        frames = [(agent, agent.generate_signal_series(df).reindex(common)) for agent in agents]
        # Number of history rows up to each common date (generate_signal is
        # skipped for prefixes shorter than 2 rows).
        n_rows = pd.Series(range(1, len(df) + 1), index=df.index).reindex(common)
        per_symbol[symbol] = (frames, n_rows.to_numpy())

    for pos, date in enumerate(common):
        by_symbol: Dict[str, List[AgentSignal]] = {}
        for symbol, (frames, n_rows) in per_symbol.items():
            if n_rows[pos] < 2:
                continue
            sigs: List[AgentSignal] = []
            for agent, frame in frames:
                sig = int(frame["signal"].iat[pos])
                conf = float(frame["confidence"].iat[pos])
                sigs.append({
                    "symbol": symbol,
                    "agent": agent.name,
                    "signal": sig,
                    "confidence": conf,
                    "kind": agent.kind,
                })
            by_symbol[symbol] = sigs
        signals[date] = by_symbol
    return signals


def _prices_on(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    date: pd.Timestamp,
    column: str,
) -> Dict[str, float]:
    return {
        symbol: float(data_by_symbol[symbol].at[date, column])
        for symbol in symbols
        if date in data_by_symbol[symbol].index
    }


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
    fill: FillMode = "close",
) -> PortfolioSimulator:
    """
    Run a daily multi-agent backtest.

    If `precomputed_signals` is provided, per-day signals are reused from that
    table; otherwise they are computed here. `start_date` and `end_date` bound
    the trading window (inclusive). `agent_filter`, when given, keeps only
    signals whose `agent` field matches one of the listed names (useful for
    ablation and single-agent baselines).

    `fill` selects when decisions made on day t are executed (see `FillMode`).
    With "next_open" equity is still marked at each day's Close, and decisions
    from the last day of the window are not executed.
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
    if fill not in ("close", "next_open"):
        raise ValueError(f"Unknown fill mode: {fill!r}")

    decision_agent = decision_agent or DecisionAgent(mode="aggressive")
    portfolio = PortfolioSimulator(**simulator_kwargs)

    if precomputed_signals is None:
        precomputed_signals = precompute_signals(agents, data_by_symbol)

    symbols = list(data_by_symbol.keys())
    dates = _common_dates(data_by_symbol)
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]

    allowed_agents = set(agent_filter) if agent_filter is not None else None
    pending: List[Decision] = []

    for date in dates:
        if fill == "next_open" and pending:
            portfolio.execute(date, _prices_on(data_by_symbol, symbols, date, "Open"), pending)
            pending = []

        by_symbol = precomputed_signals.get(date, {})
        daily_signals: List[AgentSignal] = [
            sig
            for symbol in symbols
            for sig in by_symbol.get(symbol, [])
            if allowed_agents is None or sig["agent"] in allowed_agents
        ]
        decisions = decision_agent.decide(daily_signals)
        close_prices = _prices_on(data_by_symbol, symbols, date, "Close")

        if fill == "close":
            portfolio.execute(date, close_prices, decisions)
        else:
            pending = decisions
        portfolio.mark(date, close_prices)

    return portfolio
