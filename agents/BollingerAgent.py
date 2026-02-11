from agents.base.BaseAgent import BaseAgent
from agents.base.AgentSignal import AgentSignal
import pandas as pd


class BollingerAgent(BaseAgent):
    name = "bollinger"
    horizon = "daily"
    kind = "mean_reversion"

    def __init__(self, window=20, std_mult=2):
        self.window = window
        self.std_mult = std_mult

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.window:
            return self._neutral_signal(symbol)

        close = data["Close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()

        upper = ma + self.std_mult * std
        lower = ma - self.std_mult * std
        last = close.iloc[-1]

        if last < lower.iloc[-1]:
            return self._pack(symbol, 1, (lower.iloc[-1] - last) / last)
        if last > upper.iloc[-1]:
            return self._pack(symbol, -1, (last - upper.iloc[-1]) / last)

        return self._neutral_signal(symbol)
