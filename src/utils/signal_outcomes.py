"""
Pooled signal-outcome table used by the confidence-validation and
inverse-variance-weight experiments.

Each row pairs one active agent signal at date t with the realized forward
return of that symbol over the following `horizon` trading days, so that the
directional correctness of the signal can be evaluated ex post. Rows are
restricted to dates whose full forward horizon lies inside the requested
window, which keeps the table free of label leakage across the train/test
boundary.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.backtesting import SignalsBySymbolByDate


OUTCOME_COLUMNS = [
    "date",
    "symbol",
    "agent",
    "kind",
    "signal",
    "confidence",
    "forward_return",
    "correct",
]


def build_signal_outcome_frame(
    signals: SignalsBySymbolByDate,
    data_by_symbol: Dict[str, pd.DataFrame],
    horizon: int = 1,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of active signals paired with realized forward returns.

    Parameters
    ----------
    signals : SignalsBySymbolByDate
        Precomputed per-date, per-symbol agent signals.
    data_by_symbol : dict
        Symbol -> daily OHLCV DataFrame (Close column required).
    horizon : int
        Forward horizon in trading days over which correctness is judged.
    end : pd.Timestamp, optional
        If given, only signals whose forward horizon ends on or before this
        date are kept (use the train-end boundary to stay in-sample).

    Returns
    -------
    pd.DataFrame with OUTCOME_COLUMNS. Only active signals (signal != 0) are
    included; `correct` is 1 when the sign of the forward return matches the
    signal and 0 otherwise.
    """
    closes: Dict[str, pd.Series] = {
        sym: df["Close"].astype(float) for sym, df in data_by_symbol.items()
    }
    positions: Dict[str, Dict[pd.Timestamp, int]] = {
        sym: {date: i for i, date in enumerate(close.index)}
        for sym, close in closes.items()
    }

    rows: List[dict] = []
    for date, by_symbol in signals.items():
        for symbol, symbol_signals in by_symbol.items():
            close = closes.get(symbol)
            if close is None:
                continue
            pos = positions[symbol].get(date)
            if pos is None or pos + horizon >= len(close):
                continue
            target_date = close.index[pos + horizon]
            if end is not None and target_date > pd.Timestamp(end):
                continue
            fwd = float(close.iloc[pos + horizon] / close.iloc[pos] - 1.0)
            for sig in symbol_signals:
                if sig["signal"] == 0:
                    continue
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "agent": sig["agent"],
                    "kind": sig["kind"],
                    "signal": int(sig["signal"]),
                    "confidence": float(sig["confidence"]),
                    "forward_return": fwd,
                    "correct": int(np.sign(fwd) == sig["signal"]),
                })

    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
