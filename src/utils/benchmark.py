from pathlib import Path
from typing import List, Optional

import pandas as pd

WIG20_FILENAME = "wig20.csv"


def build_buy_and_hold_curve(
    equity_curve: List[dict],
    data_dir: Path,
    initial_cash: float,
    index_filename: str = WIG20_FILENAME,
    date_col: str = "Date",
) -> Optional[List[dict]]:
    """
    Build a frictionless buy-and-hold equity curve for the market index over
    the same dates as the strategy.

    The benchmark invests the full initial cash into the index on the first
    date and holds it untouched; no commission, slippage, or tax is charged,
    matching the passive benchmark convention used throughout the evaluation.
    Returns a list of {date, equity} or None if the index data is unavailable.
    """
    path = data_dir / index_filename
    if not path.exists():
        return None
    try:
        index = pd.read_csv(path, parse_dates=[date_col])
        if "Close" not in index.columns:
            return None
        index = index.dropna(subset=[date_col, "Close"]).set_index(date_col).sort_index()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return None

    strategy_dates = [e["date"] for e in equity_curve]
    if not strategy_dates:
        return None

    # Align index close to strategy dates (forward-fill if missing)
    close = index["Close"].reindex(strategy_dates, method="ffill")
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


def build_buy_and_hold_wig20_curve(
    equity_curve: List[dict],
    data_dir: Path,
    initial_cash: float,
    date_col: str = "Date",
) -> Optional[List[dict]]:
    """Backward-compatible wrapper around `build_buy_and_hold_curve` for WIG20."""
    return build_buy_and_hold_curve(
        equity_curve, data_dir, initial_cash, WIG20_FILENAME, date_col
    )


def buy_and_hold_total_return(
    benchmark_curve: List[dict], initial_cash: float
) -> Optional[float]:
    """Return total return (fraction) for a buy-and-hold curve, or None if invalid."""
    if not benchmark_curve or initial_cash <= 0:
        return None
    final_equity = benchmark_curve[-1]["equity"]
    return (final_equity - initial_cash) / initial_cash


def buy_and_hold_wig20_total_return(
    benchmark_curve: List[dict], initial_cash: float
) -> Optional[float]:
    """Backward-compatible alias of `buy_and_hold_total_return`."""
    return buy_and_hold_total_return(benchmark_curve, initial_cash)
