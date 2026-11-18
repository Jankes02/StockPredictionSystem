"""
Daily backtest engine for a time-varying universe.

Differences from the paper's `src.utils.backtesting.backtest_portfolio_daily`
(which stays unchanged for reproducibility):
  * the trading calendar is given explicitly (e.g. the benchmark's sessions)
    instead of being the intersection of every symbol's dates, so adding a
    recently listed stock does not shorten the backtest;
  * on each day only symbols in the universe at that date may be bought;
  * a held symbol that leaves the universe is handled by `exit_policy`.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
from src.agents.base.Decision import Decision
from src.agents.DecisionAgent import DecisionAgent
from src.universe.screener import UniverseSchedule
from src.utils.PortfolioSimulator import PortfolioSimulator

FillMode = Literal["close", "next_open"]
ExitPolicy = Literal["hold_until_sell_signal", "force_exit"]


class SignalTable:
    """
    Agent signals for many symbols on a common calendar, stored as dense
    arrays (agents x calendar x symbols) instead of millions of dicts.
    A cell is "no signal" when the symbol has no bar that day or fewer than
    2 bars of history (same rule as the paper backtest).
    """

    def __init__(self, agents: Sequence[BaseAgent], data_by_symbol: Dict[str, pd.DataFrame],
                 calendar: pd.DatetimeIndex):
        self.agents = list(agents)
        self.calendar = pd.DatetimeIndex(calendar).sort_values()
        self.symbols = sorted(data_by_symbol)
        self._col = {s: j for j, s in enumerate(self.symbols)}
        shape = (len(self.agents), len(self.calendar), len(self.symbols))
        self.signal = np.zeros(shape, dtype=np.int8)
        self.confidence = np.zeros(shape, dtype=np.float64)
        self.present = np.zeros(shape[1:], dtype=bool)

        for j, s in enumerate(self.symbols):
            df = data_by_symbol[s]
            rows = self.calendar.get_indexer(df.index)   # -1 for bars off-calendar
            ok = (rows >= 0) & (np.arange(len(df)) >= 1)
            self.present[rows[ok], j] = True
            for a, agent in enumerate(self.agents):
                frame = agent.generate_signal_series(df)
                self.signal[a, rows[ok], j] = frame["signal"].to_numpy()[ok]
                self.confidence[a, rows[ok], j] = frame["confidence"].to_numpy()[ok]

    def row(self, date: pd.Timestamp) -> int:
        i = self.calendar.get_loc(pd.Timestamp(date))
        return int(i)

    def signals_on(self, date: pd.Timestamp, symbols: Sequence[str]) -> List[AgentSignal]:
        i = self.row(date)
        out: List[AgentSignal] = []
        for s in symbols:
            j = self._col.get(s)
            if j is None or not self.present[i, j]:
                continue
            for a, agent in enumerate(self.agents):
                out.append({
                    "symbol": s,
                    "agent": agent.name,
                    "signal": int(self.signal[a, i, j]),
                    "confidence": float(self.confidence[a, i, j]),
                    "kind": agent.kind,
                })
        return out


@dataclass
class PriceTable:
    """Open/Close lookups on the calendar (NaN where a symbol has no bar)."""
    open: pd.DataFrame
    close: pd.DataFrame

    @classmethod
    def build(cls, data_by_symbol: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> "PriceTable":
        cal = pd.DatetimeIndex(calendar)
        return cls(
            open=pd.DataFrame({s: d["Open"] for s, d in data_by_symbol.items()}).reindex(cal),
            close=pd.DataFrame({s: d["Close"] for s, d in data_by_symbol.items()}).reindex(cal),
        )

    @staticmethod
    def _row(frame: pd.DataFrame, date) -> Dict[str, float]:
        r = frame.loc[date]
        r = r[r.notna()]
        return {k: float(v) for k, v in r.items()}

    def opens(self, date) -> Dict[str, float]:
        return self._row(self.open, date)

    def closes(self, date) -> Dict[str, float]:
        return self._row(self.close, date)


def run_daily_backtest(
    *,
    signals: SignalTable,
    prices: PriceTable,
    universe: UniverseSchedule,
    decision_agent: DecisionAgent,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    fill: FillMode = "next_open",
    exit_policy: ExitPolicy = "hold_until_sell_signal",
    agent_filter: Optional[Sequence[str]] = None,
    initial_cash: float = 100_000,
    position_size: float = 0.1,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    stamp_duty_bps: float = 0.0,
) -> PortfolioSimulator:
    if fill not in ("close", "next_open"):
        raise ValueError(f"Unknown fill mode: {fill!r}")
    if exit_policy not in ("hold_until_sell_signal", "force_exit"):
        raise ValueError(f"Unknown exit policy: {exit_policy!r}")

    portfolio = PortfolioSimulator(
        initial_cash=initial_cash, position_size=position_size,
        commission_bps=commission_bps, slippage_bps=slippage_bps, stamp_duty_bps=stamp_duty_bps,
    )
    dates = signals.calendar
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    allowed = set(agent_filter) if agent_filter is not None else None
    pending: List[Decision] = []

    for date in dates:
        if fill == "next_open" and pending:
            portfolio.execute(date, prices.opens(date), pending)
            pending = []

        members = universe.members_on(date)
        held = set(portfolio.positions)
        # Held non-members keep being evaluated so a SELL signal can close them.
        candidates = sorted(members | held)
        day_signals = [
            s for s in signals.signals_on(date, candidates)
            if allowed is None or s["agent"] in allowed
        ]
        decisions = [
            d for d in decision_agent.decide(day_signals)
            if d["action"] != "BUY" or d["symbol"] in members
        ]
        if exit_policy == "force_exit":
            selling = {d["symbol"] for d in decisions if d["action"] == "SELL"}
            for sym in sorted(held - members - selling):
                decisions.append({"symbol": sym, "action": "SELL", "score": 0.0,
                                  "confidence": 0.0, "contributing_agents": []})

        close = prices.closes(date)
        if fill == "close":
            portfolio.execute(date, close, decisions)
        else:
            pending = decisions
        portfolio.mark(date, close)

    return portfolio
