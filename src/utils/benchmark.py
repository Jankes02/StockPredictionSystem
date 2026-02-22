from pathlib import Path
from typing import List, Optional

import pandas as pd

WIG20_FILENAME = "wig20.csv"


def build_buy_and_hold_wig20_curve(
    equity_curve: List[dict],
    data_dir: Path,
    initial_cash: float,
    date_col: str = "Date",
) -> Optional[List[dict]]:
    """
    Build equity curve for buy-and-hold WIG20 over the same dates as the strategy.
    Returns list of {date, equity} or None if wig20 data is not available.
    """
    path = data_dir / WIG20_FILENAME
    if not path.exists():
        return None
    try:
        wig = pd.read_csv(path, parse_dates=[date_col])
        if "Close" not in wig.columns:
            return None
        wig = wig.dropna(subset=[date_col, "Close"]).set_index(date_col).sort_index()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return None

    strategy_dates = [e["date"] for e in equity_curve]
    if not strategy_dates:
        return None

    # Align WIG20 close to strategy dates (forward-fill if missing)
    close = wig["Close"].reindex(strategy_dates, method="ffill")
    close = close.dropna()
    if close.empty:
        return None

    first_close = float(close.iloc[0])
    if first_close <= 0:
        return None

    benchmark_curve = [
        {"date": d, "equity": initial_cash * (float(close.loc[d]) / first_close)}
        for d in close.index
    ]
    return benchmark_curve


def buy_and_hold_wig20_total_return(
    benchmark_curve: List[dict], initial_cash: float
) -> Optional[float]:
    """Return total return (fraction) for buy-and-hold WIG20, or None if invalid."""
    if not benchmark_curve or initial_cash <= 0:
        return None
    final_equity = benchmark_curve[-1]["equity"]
    return (final_equity - initial_cash) / initial_cash
