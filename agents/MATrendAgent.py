from agents.base.AgentSignal import AgentSignal
from agents.base.BaseAgent import BaseAgent
import pandas as pd


class MATrendAgent(BaseAgent):
    name = "ma_trend"
    horizon = "daily"
    kind = "trend"

    def __init__(self, fast=50, slow=200):
        self.fast = fast
        self.slow = slow

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.slow:
            return self._neutral_signal(symbol)

        close = data["Close"]
        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()

        diff = ma_fast.iloc[-1] - ma_slow.iloc[-1]
        last_close = close.iloc[-1]
        if pd.isna(last_close) or last_close <= 0:
            return self._neutral_signal(symbol)
        signal = 1 if diff > 0 else -1
        confidence = abs(diff) / last_close

        return self._pack(symbol, signal, confidence)
