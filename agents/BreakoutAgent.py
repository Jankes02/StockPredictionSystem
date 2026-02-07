from base.BaseAgent import BaseAgent
from base.AgentSignal import AgentSignal
import pandas as pd


class BreakoutAgent(BaseAgent):
    name = "breakout"
    horizon = "intraday"
    kind = "momentum"

    def __init__(self, lookback=20):
        self.lookback = lookback

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.lookback + 1:
            return self._neutral_signal(symbol)

        high = data["High"].iloc[-self.lookback:-1].max()
        low = data["Low"].iloc[-self.lookback:-1].min()
        close = data["Close"].iloc[-1]

        if close > high:
            return self._pack(symbol, 1, (close - high) / high)
        if close < low:
            return self._pack(symbol, -1, (low - close) / low)

        return self._neutral_signal(symbol)
