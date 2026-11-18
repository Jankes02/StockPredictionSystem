"""
Liquidity screener and point-in-time universe schedule.

A snapshot taken "as of" session D uses only bars up to and including D and
becomes effective on the next session, so a backtest over the schedule never
trades a stock because of liquidity it had in the future.

Ranking: median daily dollar volume (Close x Volume) over `lookback` sessions.
Eligibility at D:
  * close at D >= min_price,
  * at least `min_bars` bars within the last `lookback` sessions of the calendar
    (excludes halted / recently listed / sparsely traded names),
  * at least `min_history` bars in total (the 200-day MA agent needs them).
Hysteresis: a current member stays while its rank is within
`top_n * keep_buffer`, which cuts turnover caused by small rank changes.
"""
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenCriteria:
    top_n: int = 100
    min_price: float = 5.0
    lookback: int = 60
    min_bars: int = 50
    min_history: int = 200
    keep_buffer: float = 1.25


@dataclass
class Snapshot:
    as_of: pd.Timestamp
    effective: Optional[pd.Timestamp]   # None if no session after as_of yet
    table: pd.DataFrame                  # index symbol; rank, median_dollar_volume, close

    @property
    def members(self) -> List[str]:
        return self.table.index.tolist()


class LiquidityPanel:
    """Close and dollar-volume panels (calendar x symbols) for the whole pool."""

    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex):
        self.calendar = pd.DatetimeIndex(calendar).sort_values()
        symbols = sorted(data_by_symbol)
        close = {s: data_by_symbol[s]["Close"] for s in symbols}
        dv = {s: data_by_symbol[s]["Close"] * data_by_symbol[s]["Volume"] for s in symbols}
        self.close = pd.DataFrame(close).reindex(self.calendar).astype("float32")
        self.dollar_volume = pd.DataFrame(dv).reindex(self.calendar).astype("float32")
        # bars available up to each date (history length), from each symbol's own index
        self.history = pd.DataFrame(
            {s: pd.Series(np.arange(1, len(data_by_symbol[s]) + 1), index=data_by_symbol[s].index)
             for s in symbols}
        ).reindex(self.calendar).ffill().fillna(0).astype("int32")

    @property
    def symbols(self) -> List[str]:
        return self.close.columns.tolist()

    def screen(self, as_of: pd.Timestamp, criteria: ScreenCriteria,
               previous: Optional[Set[str]] = None) -> pd.DataFrame:
        pos = self.calendar.searchsorted(pd.Timestamp(as_of), side="right") - 1
        if pos < 0:
            return _empty_table()
        lo = max(0, pos - criteria.lookback + 1)
        window = self.dollar_volume.iloc[lo: pos + 1].to_numpy(dtype="float64")
        n_bars = np.sum(~np.isnan(window), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            median_dv = np.nanmedian(window, axis=0)
        close = self.close.iloc[pos].to_numpy(dtype="float64")
        hist = self.history.iloc[pos].to_numpy()

        eligible = (close >= criteria.min_price) & (n_bars >= criteria.min_bars) & \
                   (hist >= criteria.min_history) & ~np.isnan(median_dv)
        ranked = pd.DataFrame({"median_dollar_volume": median_dv, "close": close},
                              index=self.close.columns)[eligible]
        ranked = ranked.sort_values("median_dollar_volume", ascending=False, kind="mergesort")
        ranked["rank"] = np.arange(1, len(ranked) + 1)

        top = ranked[ranked["rank"] <= criteria.top_n]
        if previous:
            keep_rank = int(np.floor(criteria.top_n * criteria.keep_buffer))
            kept = ranked[(ranked["rank"] <= keep_rank) & ranked.index.isin(list(previous))]
            # members kept by the buffer displace the lowest-ranked newcomers
            newcomers = top[~top.index.isin(kept.index)]
            n_new = max(0, criteria.top_n - len(kept))
            top = pd.concat([kept, newcomers.head(n_new)]).sort_values("rank")
        out = top[["rank", "median_dollar_volume", "close"]].copy()
        out.index.name = "symbol"
        return out


def _empty_table() -> pd.DataFrame:
    t = pd.DataFrame(columns=["rank", "median_dollar_volume", "close"])
    t.index.name = "symbol"
    return t


def month_end_sessions(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of each calendar month present in `calendar`."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex(s.groupby([cal.year, cal.month]).max().to_numpy())


def build_universe_history(
    panel: LiquidityPanel,
    criteria: ScreenCriteria,
    rebalance_dates: Sequence[pd.Timestamp],
) -> List[Snapshot]:
    """Sequential snapshots (hysteresis depends on the previous membership)."""
    snapshots: List[Snapshot] = []
    previous: Optional[Set[str]] = None
    cal = panel.calendar
    for d in sorted(pd.Timestamp(x) for x in rebalance_dates):
        table = panel.screen(d, criteria, previous)
        nxt = cal[cal > d]
        snapshots.append(Snapshot(as_of=d, effective=nxt[0] if len(nxt) else None, table=table))
        previous = set(table.index)
    return snapshots


class UniverseSchedule:
    """Membership as a step function of time, from snapshots."""

    def __init__(self, snapshots: Sequence[Snapshot]):
        snaps = sorted((s for s in snapshots if s.effective is not None), key=lambda s: s.effective)
        self._effective = [s.effective for s in snaps]
        self._members = [frozenset(s.members) for s in snaps]

    @classmethod
    def static(cls, symbols: Sequence[str], start: pd.Timestamp) -> "UniverseSchedule":
        table = pd.DataFrame(index=pd.Index(list(symbols), name="symbol"))
        return cls([Snapshot(as_of=start, effective=pd.Timestamp(start), table=table)])

    def members_on(self, date: pd.Timestamp) -> frozenset:
        i = bisect_right(self._effective, pd.Timestamp(date)) - 1
        return self._members[i] if i >= 0 else frozenset()

    def all_symbols(self) -> List[str]:
        return sorted(set().union(*self._members)) if self._members else []

    @property
    def first_effective(self) -> Optional[pd.Timestamp]:
        return self._effective[0] if self._effective else None
