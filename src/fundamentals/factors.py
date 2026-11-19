"""
Point-in-time fundamental factors and their cross-sectional scores.

Raw factors (as known on a date, see edgar.point_in_time):
  gpa        gross profit / assets            (profitability, Novy-Marx)
  roa        net income / assets
  cfoa       operating cash flow / assets
  leverage   liabilities / assets             (lower is better)
  accruals   (net income - operating cash flow) / assets (lower is better)
  ey         net income / market cap          (earnings yield)
  bm         equity / market cap              (book-to-market)
Scores are average cross-sectional percentile ranks among the symbols being
compared on that date: quality = gpa, roa, cfoa, -leverage, -accruals;
value = ey, bm; quality_value = both groups with equal weight.
Symbols without the needed filings get no score (they are never selected).
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.fundamentals.edgar import FundamentalsStore, point_in_time

QUALITY = {"gpa": 1, "roa": 1, "cfoa": 1, "leverage": -1, "accruals": -1}
VALUE = {"ey": 1, "bm": 1}


class FactorPanel:
    def __init__(self, store: FundamentalsStore, symbols: Iterable[str], dates: Sequence[pd.Timestamp],
                 close: pd.DataFrame):
        """`dates`: sessions on which scores will be asked for (e.g. month-ends);
        `close`: calendar x symbols closes for market caps."""
        self.dates = pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in dates)))
        self.close = close
        self._raw: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            facts = store.read(s)
            if facts is None or facts.empty:
                continue
            self._raw[s] = point_in_time(facts, self.dates)

    @property
    def symbols(self) -> List[str]:
        return sorted(self._raw)

    def factors(self, date, symbols: Sequence[str]) -> pd.DataFrame:
        d = pd.Timestamp(date)
        if d not in self.dates:
            d = self.dates[self.dates <= d][-1] if (self.dates <= d).any() else None
        rows = {}
        px = self.close.loc[pd.Timestamp(date)] if pd.Timestamp(date) in self.close.index else None
        for s in symbols:
            raw = self._raw.get(s)
            if raw is None or d is None:
                continue
            v = raw.loc[d]
            assets = v.get("assets")
            if not assets or not np.isfinite(assets) or assets <= 0:
                continue
            price = px.get(s) if px is not None else np.nan
            mcap = v.get("shares", np.nan) * price if price is not None else np.nan
            rows[s] = {
                "gpa": v.get("gross_profit", np.nan) / assets,
                "roa": v.get("net_income", np.nan) / assets,
                "cfoa": v.get("operating_cash_flow", np.nan) / assets,
                "leverage": v.get("liabilities", np.nan) / assets,
                "accruals": (v.get("net_income", np.nan) - v.get("operating_cash_flow", np.nan)) / assets,
                "ey": v.get("net_income", np.nan) / mcap if mcap and mcap > 0 else np.nan,
                "bm": v.get("equity", np.nan) / mcap if mcap and mcap > 0 else np.nan,
            }
        return pd.DataFrame.from_dict(rows, orient="index", columns=list(QUALITY) + list(VALUE))

    @staticmethod
    def _score(f: pd.DataFrame, weights: Dict[str, int], min_factors: int) -> pd.Series:
        ranks = []
        for col, sign in weights.items():
            x = f[col] * sign
            ranks.append(x.rank(pct=True))
        r = pd.concat(ranks, axis=1)
        score = r.mean(axis=1, skipna=True)
        return score[r.notna().sum(axis=1) >= min_factors]

    def score(self, date, symbols: Sequence[str], kind: str = "quality") -> pd.Series:
        f = self.factors(date, symbols)
        if f.empty:
            return pd.Series(dtype=float)
        if kind == "quality":
            return self._score(f, QUALITY, 3)
        if kind == "value":
            return self._score(f, VALUE, 1)
        if kind == "quality_value":
            q, v = self._score(f, QUALITY, 3), self._score(f, VALUE, 1)
            both = pd.concat([q, v], axis=1)
            return both.mean(axis=1)[both.notna().all(axis=1)]
        raise ValueError(f"Unknown score kind {kind!r}")


def quality_filtered_schedule(schedule, snapshots, panel: FactorPanel, keep_fraction: float,
                              kind: str = "quality"):
    """
    New UniverseSchedule keeping, in each snapshot, the top `keep_fraction` of
    members by `kind` score as of the snapshot's screening date (members
    without a score are dropped).
    """
    from src.universe.screener import Snapshot, UniverseSchedule

    out = []
    for snap in snapshots:
        members = snap.members
        if not members:
            out.append(snap)
            continue
        sc = panel.score(snap.as_of, members, kind).sort_values(ascending=False, kind="mergesort")
        keep = list(sc.index[: max(1, int(round(len(members) * keep_fraction)))]) if len(sc) else []
        out.append(Snapshot(as_of=snap.as_of, effective=snap.effective, table=snap.table.loc[keep]))
    return UniverseSchedule(out)
