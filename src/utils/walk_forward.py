"""
Walk-forward evaluation for the multi-agent trading system.

Indicator hyperparameters are frozen at their literature-conventional defaults
throughout. Only the decision-agent parameters (`mode`, `min_confidence`) are
re-fit per training window. See the IEEE Access paper, Section V-C.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.utils.backtesting import (
    SignalsBySymbolByDate,
    backtest_portfolio_daily,
    precompute_signals,
)
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.PortfolioSimulator import PortfolioSimulator


MODES = ("passive", "balanced", "aggressive")
DEFAULT_MIN_CONFIDENCES: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
SELECTION_METRIC = "total_return"


@dataclass
class EvaluationResult:
    """Metrics for a single (train, test) evaluation run."""
    mode: str
    min_confidence: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FoldResult:
    """Outcome of one anchored walk-forward fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_mode: str
    selected_min_confidence: float
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_portfolio: Optional[PortfolioSimulator] = None


class WalkForwardEvaluator:
    """
    Evaluate the decision-agent configuration on held-out data.

    Two protocols are provided:
      * `evaluate_anchored_split`: single anchored split (headline protocol).
      * `evaluate_k_folds`: K=3 anchored walk-forward (robustness appendix).
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        data_by_symbol: Dict[str, pd.DataFrame],
        simulator_kwargs: Optional[Dict] = None,
        modes: Sequence[str] = MODES,
        min_confidences: Sequence[float] = DEFAULT_MIN_CONFIDENCES,
        selection_metric: str = SELECTION_METRIC,
    ):
        self.agents = list(agents)
        self.data_by_symbol = data_by_symbol
        self.simulator_kwargs = simulator_kwargs or {}
        self.modes = list(modes)
        self.min_confidences = list(min_confidences)
        self.selection_metric = selection_metric

        # Signals are computed once on the full history and reused for every
        # grid point and every train/test slice. This is leakage-safe because
        # each signal at date t depends only on data up to t.
        self.signals: SignalsBySymbolByDate = precompute_signals(
            self.agents, self.data_by_symbol
        )

    def evaluate_anchored_split(
        self,
        train_end: pd.Timestamp,
        *,
        train_start: Optional[pd.Timestamp] = None,
        test_end: Optional[pd.Timestamp] = None,
    ) -> FoldResult:
        """Run grid search on train window, evaluate best config on held-out test."""
        dates = self._common_dates()
        train_start = pd.Timestamp(train_start) if train_start else dates[0]
        train_end = pd.Timestamp(train_end)
        test_start = self.first_date_after(train_end)
        test_end = pd.Timestamp(test_end) if test_end else dates[-1]

        grid = self.run_grid_search(train_start, train_end)
        best = self._pick_best(grid)

        test_portfolio = self.run(
            mode=best.mode,
            min_confidence=best.min_confidence,
            start=test_start,
            end=test_end,
        )
        test_metrics = MetricsCalculator(
            test_portfolio.equity_curve, test_portfolio.trades
        ).summary()

        return FoldResult(
            fold_index=0,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            selected_mode=best.mode,
            selected_min_confidence=best.min_confidence,
            train_metrics=best.train_metrics,
            test_metrics=test_metrics,
            test_portfolio=test_portfolio,
        )

    def evaluate_k_folds(
        self,
        first_test_start: pd.Timestamp,
        *,
        folds: int = 3,
        train_start: Optional[pd.Timestamp] = None,
        last_test_end: Optional[pd.Timestamp] = None,
    ) -> List[FoldResult]:
        """
        Anchored walk-forward with expanding train windows.

        The out-of-sample period [first_test_start, last_test_end] is split
        into `folds` contiguous test slices. Fold k trains on everything
        strictly before its test slice and evaluates on that slice.
        """
        dates = self._common_dates()
        first_test_start = pd.Timestamp(first_test_start)
        last_test_end = pd.Timestamp(last_test_end) if last_test_end else dates[-1]
        train_start = pd.Timestamp(train_start) if train_start else dates[0]

        test_dates = dates[(dates >= first_test_start) & (dates <= last_test_end)]
        if len(test_dates) < folds:
            raise ValueError(
                f"Not enough test dates ({len(test_dates)}) for {folds} folds"
            )

        bucket_edges = [
            test_dates[int(round(i * len(test_dates) / folds))]
            for i in range(folds)
        ] + [test_dates[-1]]

        results: List[FoldResult] = []
        for k in range(folds):
            slice_start = bucket_edges[k]
            slice_end = (
                bucket_edges[k + 1]
                if k == folds - 1
                else self.prev_date_before(bucket_edges[k + 1])
            )
            fold_train_end = self.prev_date_before(slice_start)

            grid = self.run_grid_search(train_start, fold_train_end)
            best = self._pick_best(grid)

            test_portfolio = self.run(
                mode=best.mode,
                min_confidence=best.min_confidence,
                start=slice_start,
                end=slice_end,
            )
            test_metrics = MetricsCalculator(
                test_portfolio.equity_curve, test_portfolio.trades
            ).summary()

            results.append(
                FoldResult(
                    fold_index=k,
                    train_start=train_start,
                    train_end=fold_train_end,
                    test_start=slice_start,
                    test_end=slice_end,
                    selected_mode=best.mode,
                    selected_min_confidence=best.min_confidence,
                    train_metrics=best.train_metrics,
                    test_metrics=test_metrics,
                    test_portfolio=test_portfolio,
                )
            )
        return results

    # ---------- PUBLIC HELPERS ----------

    def run_grid_search(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> List[EvaluationResult]:
        """Backtest every (mode, min_confidence) combination over the window."""
        grid: List[EvaluationResult] = []
        for mode in self.modes:
            for mc in self.min_confidences:
                portfolio = self.run(
                    mode=mode,
                    min_confidence=mc,
                    start=start,
                    end=end,
                )
                metrics = MetricsCalculator(
                    portfolio.equity_curve, portfolio.trades
                ).summary()
                grid.append(EvaluationResult(
                    mode=mode,
                    min_confidence=mc,
                    train_start=start,
                    train_end=end,
                    test_start=start,
                    test_end=end,
                    train_metrics=metrics,
                ))
        return grid

    def run(
        self,
        *,
        mode: str,
        min_confidence: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
        agent_filter: Optional[Sequence[str]] = None,
    ) -> PortfolioSimulator:
        """Single backtest with the given decision-agent configuration."""
        decision_agent = DecisionAgent(mode=mode, min_confidence=min_confidence)
        return backtest_portfolio_daily(
            agents=self.agents,
            data_by_symbol=self.data_by_symbol,
            decision_agent=decision_agent,
            precomputed_signals=self.signals,
            start_date=start,
            end_date=end,
            agent_filter=agent_filter,
            **self.simulator_kwargs,
        )

    def common_dates(self) -> pd.DatetimeIndex:
        return self._common_dates()

    def first_date_after(self, date: pd.Timestamp) -> pd.Timestamp:
        dates = self._common_dates()
        after = dates[dates > pd.Timestamp(date)]
        if len(after) == 0:
            raise ValueError(f"No common dates after {date}")
        return after[0]

    def prev_date_before(self, date: pd.Timestamp) -> pd.Timestamp:
        dates = self._common_dates()
        before = dates[dates < pd.Timestamp(date)]
        if len(before) == 0:
            raise ValueError(f"No common dates before {date}")
        return before[-1]

    # ---------- INTERNAL ----------

    def _pick_best(self, grid: Sequence[EvaluationResult]) -> EvaluationResult:
        return max(
            grid,
            key=lambda r: r.train_metrics.get(self.selection_metric, float("-inf")),
        )

    def _common_dates(self) -> pd.DatetimeIndex:
        symbols = list(self.data_by_symbol.keys())
        common = self.data_by_symbol[symbols[0]].index
        for sym in symbols[1:]:
            common = common.intersection(self.data_by_symbol[sym].index)
        return common.sort_values()
