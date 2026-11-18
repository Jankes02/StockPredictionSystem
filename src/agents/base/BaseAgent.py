from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.agents.base.AgentSignal import AgentSignal


class BaseAgent(ABC):
    name: str
    kind: str

    def _neutral_signal(self, symbol: str) -> AgentSignal:
        return {
            "symbol": symbol,
            "agent": self.name,
            "signal": 0,
            "confidence": 0,
            "kind": self.kind
        }

    def _pack(self, symbol: str, signal: int, confidence: float) -> AgentSignal:
        return {
            "symbol": symbol,
            "agent": self.name,
            "signal": signal,
            "confidence": min(max(confidence, 0.0), 1.0),
            "kind": self.kind
        }

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        """Signal for the last row of `data` (uses only rows up to and including it)."""
        raise NotImplementedError

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Signal for every row of `data`, as if `generate_signal(data.iloc[:i+1])`
        had been called for each i. Returns a frame indexed like `data` with
        integer column `signal` and float column `confidence`.

        All built-in indicators are causal, so subclasses override this with a
        single vectorised pass. This default is the slow reference fallback for
        agents that do not.
        """
        rows = [self.generate_signal(data.iloc[: i + 1], "") for i in range(len(data))]
        return pd.DataFrame(
            {
                "signal": [int(r["signal"]) for r in rows],
                "confidence": [float(r["confidence"]) for r in rows],
            },
            index=data.index,
        )

    @staticmethod
    def _series_frame(
        index: pd.Index,
        signal: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
    ) -> pd.DataFrame:
        """
        Assemble a signal frame, mirroring `_neutral_signal` / `_pack`:
        rows that are not `valid` or have signal 0 get confidence 0, the rest
        have confidence clipped to [0, 1] (NaN is preserved, as `_pack` does).
        """
        signal = np.where(valid, signal, 0).astype(int)
        conf = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
        conf = np.where(signal != 0, conf, 0.0)
        return pd.DataFrame({"signal": signal, "confidence": conf}, index=index)
