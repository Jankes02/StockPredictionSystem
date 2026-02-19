from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
import pandas as pd


class RSIAgent(BaseAgent):
    name = "rsi"
    kind = "mean_reversion"

    def __init__(self, period: int = 14):
        self.period = period

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        if len(data) < self.period + 1:
            return self._neutral_signal(symbol)

        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = -delta.clip(upper=0).rolling(self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last = rsi.iloc[-1]

        if pd.isna(last) or not (0 <= last <= 100):
            return self._neutral_signal(symbol)
        if last < 30:
            return self._pack(symbol, 1, (30 - last) / 30)
        if last > 70:
            return self._pack(symbol, -1, (last - 70) / 30)

        return self._neutral_signal(symbol)
