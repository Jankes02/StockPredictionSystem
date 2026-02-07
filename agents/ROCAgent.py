from base.AgentSignal import AgentSignal
from base.BaseAgent import BaseAgent
import pandas as pd


class ROCAgent(BaseAgent):
    name = "roc"
    horizon = "daily"
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
