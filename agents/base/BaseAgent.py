from abc import ABC, abstractmethod
import pandas as pd
from base.AgentSignal import AgentSignal

class BaseAgent(ABC):
    name: str
    horizon: str
    kind: str

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> AgentSignal:
        pass
