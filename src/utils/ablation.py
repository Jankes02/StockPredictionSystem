"""
Ablation experiments:
  * Leave-one-agent-out: remove each signal agent in turn and rerun the backtest.
  * Single-agent-only:   run the backtest with only one signal agent active.

Both runs use the test-window (out-of-sample) configuration selected by the
walk-forward protocol, so the numbers in the paper directly support hypotheses
H2 (ensemble beats best single agent) and H3 (ensemble is robust to removal of
any single agent).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.agents.base.BaseAgent import BaseAgent
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.PortfolioSimulator import PortfolioSimulator
from src.utils.walk_forward import WalkForwardEvaluator


@dataclass
class AblationRow:
    variant: str        # e.g. "all", "no_macd", "only_macd"
    agents: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    portfolio: Optional[PortfolioSimulator] = None


class AblationRunner:
    """Run leave-one-out and single-agent-only backtests on a fixed time window."""

    def __init__(
        self,
        evaluator: WalkForwardEvaluator,
        *,
        mode: str,
        min_confidence: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        self.evaluator = evaluator
        self.mode = mode
        self.min_confidence = min_confidence
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)

        self.agent_names: List[str] = [a.name for a in evaluator.agents]

    def run_all(self) -> List[AblationRow]:
        rows: List[AblationRow] = []
        rows.append(self._run("all", self.agent_names))
        rows.extend(self._run_leave_one_out())
        rows.extend(self._run_single_agent_only())
        return rows

    def _run_leave_one_out(self) -> List[AblationRow]:
        rows: List[AblationRow] = []
        for dropped in self.agent_names:
            kept = [n for n in self.agent_names if n != dropped]
            rows.append(self._run(f"no_{dropped}", kept))
        return rows

    def _run_single_agent_only(self) -> List[AblationRow]:
        rows: List[AblationRow] = []
        for only in self.agent_names:
            rows.append(self._run(f"only_{only}", [only]))
        return rows

    def _run(self, variant: str, agent_names: Sequence[str]) -> AblationRow:
        portfolio = self.evaluator.run(
            mode=self.mode,
            min_confidence=self.min_confidence,
            start=self.start,
            end=self.end,
            agent_filter=agent_names,
        )
        metrics = MetricsCalculator(
            portfolio.equity_curve, portfolio.trades
        ).summary()
        return AblationRow(
            variant=variant,
            agents=list(agent_names),
            metrics=metrics,
            portfolio=portfolio,
        )
