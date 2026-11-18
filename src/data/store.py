"""
Local bar cache (one parquet file per symbol and timeframe) kept in sync with a
`DataProvider` by incremental updates.

Update algorithm, per symbol:
  1. No cache -> download the full history from `history_start`.
  2. Cache present -> re-download from the bar `overlap` bars before the last
     cached one. The overlapping bars must match the cache; if they do not, the
     provider has re-adjusted the history (dividend/split) and the symbol's
     full history is downloaded again. Otherwise the new bars are appended.
Running `update` twice in a row therefore adds no bars the second time.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.base import BAR_COLUMNS, PRICE_COLUMNS, DataProvider, Timeframe
from src.data.validation import validate_bars

log = logging.getLogger(__name__)

ADJUSTMENT_RTOL = 1e-6


class ParquetBarCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, symbol: str, timeframe: Timeframe) -> Path:
        return self.root / timeframe.value / f"{symbol.upper()}.parquet"

    def read(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        p = self.path(symbol, timeframe)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def write(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        p = self.path(symbol, timeframe)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(p)  # atomic: a crash never leaves a half-written file

    def symbols(self, timeframe: Timeframe) -> List[str]:
        d = self.root / timeframe.value
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


@dataclass
class SymbolUpdate:
    symbol: str
    new_bars: int = 0
    full_refetch: bool = False
    reason: str = ""
    issues: List[str] = field(default_factory=list)


@dataclass
class UpdateReport:
    timeframe: Timeframe
    updates: Dict[str, SymbolUpdate] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    requests: int = 0

    @property
    def total_new_bars(self) -> int:
        return sum(u.new_bars for u in self.updates.values())

    def summary(self) -> str:
        refetched = [s for s, u in self.updates.items() if u.full_refetch]
        with_issues = [s for s, u in self.updates.items() if u.issues]
        return (
            f"{self.timeframe.value}: {len(self.updates)} symbols, {self.total_new_bars} new bars, "
            f"{len(refetched)} full downloads, {len(self.missing)} without data, "
            f"{len(with_issues)} with data issues, {self.requests} provider calls"
        )


class MarketDataStore:
    def __init__(
        self,
        provider: DataProvider,
        cache: ParquetBarCache,
        *,
        history_start: Dict[Timeframe, pd.Timestamp],
        overlap: int = 5,
    ):
        self.provider = provider
        self.cache = cache
        self.history_start = {tf: pd.Timestamp(ts) for tf, ts in history_start.items()}
        self.overlap = overlap

    # ---------- reading ----------

    def load(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            df = self.cache.read(s, timeframe)
            if df is None:
                continue
            df = df.loc[start:end] if (start is not None or end is not None) else df
            if not df.empty:
                out[s.upper()] = df
        return out

    # ---------- updating ----------

    def update(self, symbols: Sequence[str], timeframe: Timeframe) -> UpdateReport:
        symbols = list(dict.fromkeys(s.upper() for s in symbols))
        report = UpdateReport(timeframe=timeframe)
        start_default = self.history_start[timeframe]

        full: List[str] = []
        incremental: Dict[pd.Timestamp, List[str]] = defaultdict(list)
        cached: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            df = self.cache.read(s, timeframe)
            if df is None or df.empty:
                full.append(s)
                report.updates[s] = SymbolUpdate(s, full_refetch=True, reason="not cached")
            else:
                cached[s] = df
                fetch_from = df.index[max(0, len(df) - self.overlap)]
                incremental[fetch_from].append(s)

        # Incremental downloads, batched by common start.
        for fetch_from, group in incremental.items():
            fresh = self._get(report, group, timeframe, fetch_from)
            for s in group:
                old = cached[s]
                new = fresh.get(s)
                if new is None:
                    report.updates[s] = SymbolUpdate(s, reason="no new data from provider")
                    continue
                reason = self._overlap_mismatch(old, new)
                if reason:
                    full.append(s)
                    report.updates[s] = SymbolUpdate(s, full_refetch=True, reason=reason)
                    continue
                merged = pd.concat([old, new[new.index > old.index[-1]]])
                upd = SymbolUpdate(s, new_bars=len(merged) - len(old))
                report.updates[s] = upd
                if upd.new_bars:
                    self._store(s, timeframe, merged, upd)

        # Full downloads (new symbols and re-adjusted histories).
        if full:
            fresh = self._get(report, full, timeframe, start_default)
            for s in full:
                upd = report.updates[s]
                new = fresh.get(s)
                if new is None:
                    report.missing.append(s)
                    del report.updates[s]
                    continue
                old = cached.get(s)
                upd.new_bars = len(new) - (len(old) if old is not None else 0)
                self._store(s, timeframe, new, upd)

        for s, upd in report.updates.items():
            if upd.full_refetch and upd.reason != "not cached":
                log.info("%s %s: history re-adjusted (%s), downloaded again", s, timeframe.value, upd.reason)
            for issue in upd.issues:
                log.warning("%s %s: %s", s, timeframe.value, issue)
        if report.missing:
            log.warning("%s: no data for %s", timeframe.value, ", ".join(report.missing))
        return report

    def _get(self, report: UpdateReport, symbols, timeframe, start) -> Dict[str, pd.DataFrame]:
        report.requests += 1
        return {k.upper(): v for k, v in self.provider.get_bars(symbols, timeframe, start).items()}

    def _store(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame, upd: SymbolUpdate) -> None:
        df = df.reindex(columns=BAR_COLUMNS)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        upd.issues = validate_bars(df, timeframe)
        self.cache.write(symbol, timeframe, df)

    @staticmethod
    def _overlap_mismatch(old: pd.DataFrame, new: pd.DataFrame) -> str:
        """Empty string if the re-downloaded overlap agrees with the cache."""
        common = old.index.intersection(new.index)
        if len(common) == 0:
            return "no overlap with cache"
        a = old.loc[common, PRICE_COLUMNS].to_numpy(dtype=float)
        b = new.loc[common, PRICE_COLUMNS].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=ADJUSTMENT_RTOL, atol=0.0, equal_nan=True):
            return "overlapping bars changed"
        return ""
