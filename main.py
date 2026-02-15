from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import (
    load_config,
    get_data_dir,
    build_agents,
    build_decision_agent,
    get_backtest_options,
)
from src.utils.backtesting import backtest_portfolio_daily
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.plotting import plot_equity_with_drawdown


def load_price_data(
    folder: Path,
    symbols: List[str],
    *,
    date_col: str = "Date",
    required_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV price data for symbols that have a file in the given folder.
    Symbols without a file (e.g. ZAB in daily when only 5min exists) are skipped.
    """
    if required_columns is None:
        required_columns = ["Date", "Close"]
    loaded: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = folder / f"{symbol.lower()}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=[date_col])
            if not all(c in df.columns for c in required_columns):
                continue
            df = df.dropna(subset=required_columns)
            df = df.set_index(date_col)
            loaded[symbol] = df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
            continue
    return loaded


if __name__ == "__main__":
    cfg = load_config()
    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    decision_agent = build_decision_agent(cfg)
    opts = get_backtest_options(cfg)

    data_by_symbol = load_price_data(data_dir, symbols)

    portfolio = backtest_portfolio_daily(
        agents=agents,
        data_by_symbol=data_by_symbol,
        initial_cash=opts["initial_cash"],
        position_size=opts["position_size"],
        decision_agent=decision_agent,
    )

    metrics = MetricsCalculator(
        portfolio.equity_curve,
        portfolio.trades,
    )
    summary = metrics.summary()
    print(summary)
    plot_equity_with_drawdown(portfolio.equity_curve)