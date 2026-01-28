from typing import List, Dict, Literal
from collections import defaultdict
from base.Decision import Decision


class DecisionAgent:
    def __init__(
        self,
        mode: Literal["passive", "balanced", "aggressive"] = "balanced",
        min_confidence: float = 0.15,
        max_positions: int | None = None,
    ):
        self.mode = mode
        self.min_confidence = min_confidence
        self.max_positions = max_positions

        self.kind_weights = {
            "trend": 1.3,
            "momentum": 1.0,
            "mean_reversion": 0.8,
            "volatility": 0.5,
        }

        self.thresholds = {
            "passive": {"buy": 1.2, "sell": -1.2},
            "balanced": {"buy": 0.7, "sell": -0.7},
            "aggressive": {"buy": 0.3, "sell": -0.3},
        }

    # ---------- PUBLIC API ----------

    def decide(self, signals: List[dict]) -> List[dict]:
        if not signals:
            return []

        by_symbol = self._group_by_symbol(signals)
        decisions: List[dict] = []

        for symbol, symbol_signals in by_symbol.items():
            decision = self._decide_for_symbol(symbol, symbol_signals)
            if decision["action"] != "HOLD":
                decisions.append(decision)

        return self._apply_portfolio_constraints(decisions)

    # ---------- SYMBOL LEVEL ----------

    def _decide_for_symbol(self, symbol: str, signals: List[dict]) -> dict:
        filtered = [
            s for s in signals
            if s["signal"] != 0 and s["confidence"] >= self.min_confidence
        ]

        if not filtered:
            return self._hold(symbol)

        trend_dir = self._trend_direction(signals)
        score = 0.0
        contributors = []

        for s in filtered:
            if self.mode == "passive" and trend_dir != 0:
                if s["signal"] != trend_dir:
                    continue

            weight = self.kind_weights.get(s["kind"], 1.0)
            contribution = s["signal"] * s["confidence"] * weight
            score += contribution
            contributors.append(s["agent"])

        thresholds = self.thresholds[self.mode]

        if score >= thresholds["buy"]:
            return self._decision(symbol, "BUY", score, contributors)
        elif score <= thresholds["sell"]:
            return self._decision(symbol, "SELL", score, contributors)
        else:
            return self._hold(symbol)

    # ---------- PORTFOLIO LEVEL ----------

    def _apply_portfolio_constraints(self, decisions: List[dict]) -> List[dict]:
        if self.max_positions is None:
            return decisions

        # sortuj wg siły sygnału
        decisions = sorted(
            decisions,
            key=lambda d: abs(d["score"]),
            reverse=True,
        )

        return decisions[: self.max_positions]

    # ---------- HELPERS ----------

    def _group_by_symbol(self, signals: List[dict]) -> Dict[str, List[dict]]:
        grouped = defaultdict(list)
        for s in signals:
            grouped[s["symbol"]].append(s)
        return grouped

    def _trend_direction(self, signals: List[dict]) -> int:
        trend_signals = [
            s for s in signals
            if s["kind"] == "trend" and s["signal"] != 0
        ]
        if not trend_signals:
            return 0

        total = sum(s["signal"] * s["confidence"] for s in trend_signals)
        return 1 if total > 0 else -1

    def _decision(self, symbol: str, action: str, score: float, contributors: List[str]) -> Decision:
        return {
            "symbol": symbol,
            "action": action,
            "score": score,
            "confidence": min(abs(score), 1.0),
            "contributing_agents": contributors,
        }

    def _hold(self, symbol: str) -> Decision:
        return {
            "symbol": symbol,
            "action": "HOLD",
            "score": 0.0,
            "confidence": 0.0,
            "contributing_agents": [],
        }
