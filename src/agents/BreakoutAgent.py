from src.agents.base.BaseAgent import BaseAgent
from src.agents.base.AgentSignal import AgentSignal
import pandas as pd


class BreakoutAgent(BaseAgent):
    name = "breakout"
    horizon = "intraday"
    kind = "momentum"

    def __init__(self, lookback=20, confidence_scale=25.0):
        """
        lookback: bars used for high/low range (excl. current bar).
        confidence_scale: multiplies raw breakout fraction so it reaches decision
            thresholds; e.g. 25 means a 1.2% breakout -> ~0.3 confidence (aggressive).
        """
        self.lookback = lookback
        self.confidence_scale = confidence_scale

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.lookback + 1:
            return self._neutral_signal(symbol)

        high = data["High"].iloc[-self.lookback:-1].max()
        low = data["Low"].iloc[-self.lookback:-1].min()
        close = data["Close"].iloc[-1]

        if close > high and high > 0:
            raw = (close - high) / high
            confidence = min(1.0, raw * self.confidence_scale)
            return self._pack(symbol, 1, confidence)
        if close < low and low > 0:
            raw = (low - close) / low
            confidence = min(1.0, raw * self.confidence_scale)
            return self._pack(symbol, -1, confidence)

        return self._neutral_signal(symbol)
