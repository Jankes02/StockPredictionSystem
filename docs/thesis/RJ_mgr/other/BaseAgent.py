from abc import ABC, abstractmethod
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
        raise NotImplementedError
