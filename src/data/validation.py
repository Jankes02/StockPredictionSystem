"""Sanity checks on bar data. Problems are reported, never silently fixed."""
from typing import List

import numpy as np
import pandas as pd

from src.data.base import PRICE_COLUMNS, Timeframe

MAX_DAILY_GAP_DAYS = 6          # longer calendar gaps than a long weekend + holiday
MAX_ABS_RETURN = 0.5            # |close-to-close| above this is suspicious
MAX_ZERO_VOLUME_RUN = 5         # consecutive bars without volume


def validate_bars(df: pd.DataFrame, timeframe: Timeframe) -> List[str]:
    issues: List[str] = []
    if df.empty:
        return ["no bars"]

    idx = df.index
    if idx.has_duplicates:
        issues.append(f"{int(idx.duplicated().sum())} duplicated timestamps")
    if not idx.is_monotonic_increasing:
        issues.append("index not sorted")

    prices = df[PRICE_COLUMNS]
    n_nan = int(prices.isna().any(axis=1).sum())
    if n_nan:
        issues.append(f"{n_nan} bars with missing OHLC")
    n_nonpos = int((prices <= 0).any(axis=1).sum())
    if n_nonpos:
        issues.append(f"{n_nonpos} bars with non-positive prices")
    n_hl = int((df["High"] < df["Low"]).sum())
    if n_hl:
        issues.append(f"{n_hl} bars with High < Low")

    ret = df["Close"].pct_change().abs()
    jumps = ret[ret > MAX_ABS_RETURN]
    if len(jumps):
        first = jumps.index[0]
        issues.append(f"{len(jumps)} close-to-close moves > {MAX_ABS_RETURN:.0%} (first {first})")

    if "Volume" in df:
        zero = (df["Volume"].fillna(0) <= 0).to_numpy()
        if zero.any():
            # longest run of consecutive zero-volume bars
            runs = np.diff(np.flatnonzero(np.concatenate(([True], ~zero, [True])))) - 1
            if runs.max() >= MAX_ZERO_VOLUME_RUN:
                issues.append(f"{int(runs.max())} consecutive zero-volume bars")

    if timeframe.is_daily and len(idx) > 1:
        gaps = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
        big = gaps[gaps > pd.Timedelta(days=MAX_DAILY_GAP_DAYS)]
        if len(big):
            issues.append(f"{len(big)} gaps > {MAX_DAILY_GAP_DAYS} days (largest {big.max().days}d ending {big.idxmax().date()})")

    return issues
