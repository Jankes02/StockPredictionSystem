"""
Out-of-sample checks of a frozen daily strategy on other markets/universes,
using the same engine, sources and screener as the US research but with
data that played no part in choosing the strategy.

`ExternalResearch` is a DailyResearch built from in-memory data instead of
the live config, cache and state DB (e.g. the paper's Stooq CSVs).
"""
import logging
from dataclasses import replace
from typing import Dict, Optional, Sequence

import pandas as pd

from src.research.daily_walk_forward import DailyResearch, UniverseContext
from src.universe.screener import (
    LiquidityPanel,
    ScreenCriteria,
    UniverseSchedule,
    build_universe_history,
    month_end_sessions,
)

log = logging.getLogger(__name__)


class _Universe:
    exit_policy = "hold_until_sell_signal"


class _Cfg:
    universe = _Universe()


class ExternalResearch(DailyResearch):
    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame], benchmark_close: pd.Series,
                 agents: Sequence, base_criteria: ScreenCriteria):
        self.cfg = _Cfg()
        self.agents = list(agents)
        self.data = data_by_symbol
        self.calendar = pd.DatetimeIndex(benchmark_close.index).sort_values()
        self.benchmark = benchmark_close.reindex(self.calendar).ffill()
        self.panel = LiquidityPanel(data_by_symbol, self.calendar)
        self.base_criteria = base_criteria
        self._contexts = {}

    def context(self, universe) -> UniverseContext:
        key = universe.key()
        if key not in self._contexts:
            if universe.static_symbols:
                schedule = UniverseSchedule.static(universe.static_symbols, self.calendar[0])
            else:
                overrides = {k: getattr(universe, k) for k in
                             ("max_volatility", "rank_by", "prefilter_top_n", "prefilter_skip")
                             if getattr(universe, k) is not None}
                criteria = replace(self.base_criteria, **overrides)
                snaps = build_universe_history(self.panel, criteria, month_end_sessions(self.calendar))
                schedule = UniverseSchedule(snaps)
            data = {s: self.data[s] for s in schedule.all_symbols() if s in self.data}
            self._contexts[key] = UniverseContext(schedule, data, self.calendar, self.agents)
        return self._contexts[key]

    def equal_weight_buy_and_hold(self, start, end, initial: float = 100_000):
        """Equal-weight buy-and-hold of every stock with data on `start` (no rebalancing)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        closes = pd.DataFrame({s: d["Close"] for s, d in self.data.items()}).reindex(self.calendar)
        window = closes.loc[start:end].ffill()
        first = window.iloc[0].dropna()
        rel = window[first.index] / first
        curve = rel.mean(axis=1) * initial
        return [{"date": d, "equity": float(v)} for d, v in curve.items()]
