"""
Shared backtest of a probability panel through the portfolio simulator.

Both machine-learning baselines (LSTM, GBM) reduce to a panel of per-symbol
probabilities that the h-day-ahead return is positive. This module maps such
a panel to BUY/SELL/HOLD decisions through the symmetric no-trade band used
throughout the paper and runs them through the shared `PortfolioSimulator`,
so that every baseline is accounted for identically.
"""
from typing import Dict, List, Sequence

import pandas as pd

from src.utils.PortfolioSimulator import PortfolioSimulator


def decisions_from_probs(
    probs: Dict[str, float],
    hold_band: float,
) -> List[Dict]:
    """
    Map per-symbol up-probabilities to decisions via the no-trade band:
      prob >= 0.5 + hold_band  -> BUY   (open / keep position)
      prob <= 0.5 - hold_band  -> SELL  (close position)
      otherwise                -> no decision; open positions persist
    """
    decisions: List[Dict] = []
    for symbol, prob in probs.items():
        if pd.isna(prob):
            continue
        p = float(prob)
        if p >= 0.5 + hold_band:
            decisions.append({
                "symbol": symbol,
                "action": "BUY",
                "score": p - 0.5,
                "confidence": p,
                "contributing_agents": [],
            })
        elif p <= 0.5 - hold_band:
            decisions.append({
                "symbol": symbol,
                "action": "SELL",
                "score": p - 0.5,
                "confidence": 1.0 - p,
                "contributing_agents": [],
            })
    return decisions


def backtest_prob_panel(
    panel: pd.DataFrame,
    data_by_symbol: Dict[str, pd.DataFrame],
    dates: Sequence[pd.Timestamp],
    hold_band: float,
    simulator_kwargs: Dict,
) -> PortfolioSimulator:
    """
    Run the no-trade-band decision rule over a probability panel.

    `panel` is indexed by decision date with one column per symbol. Dates
    missing from the panel are processed with no decisions so that open
    positions are still marked to market on those days.
    """
    symbols = list(data_by_symbol.keys())
    portfolio = PortfolioSimulator(**simulator_kwargs)

    for date in dates:
        if date in panel.index:
            row = panel.loc[date]
            probs = {
                sym: row[sym]
                for sym in symbols
                if sym in panel.columns
            }
            decisions = decisions_from_probs(probs, hold_band)
        else:
            decisions = []

        prices = {
            sym: float(data_by_symbol[sym].loc[date]["Close"])
            for sym in symbols
            if date in data_by_symbol[sym].index
        }
        portfolio.process_day(date, prices, decisions)

    return portfolio
