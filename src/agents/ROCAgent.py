from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
import numpy as np
import pandas as pd


class ROCAgent(BaseAgent):
    name = "roc"
    kind = "momentum"

    def __init__(self, period=10, threshold=0.02):
        self.period = period
        self.threshold = threshold

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.period + 1:
            return self._neutral_signal(symbol)

        roc = data["Close"].pct_change(self.period).iloc[-1]

        if roc > self.threshold:
            return self._pack(symbol, 1, abs(roc))
        if roc < -self.threshold:
            return self._pack(symbol, -1, abs(roc))

        return self._neutral_signal(symbol)

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        roc = data["Close"].pct_change(self.period).to_numpy(dtype=float)
        signal = np.where(roc > self.threshold, 1, np.where(roc < -self.threshold, -1, 0))
        valid = np.arange(len(roc)) + 1 >= self.period + 1
        return self._series_frame(data.index, signal, np.abs(roc), valid)
