from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
import numpy as np
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

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = -delta.clip(upper=0).rolling(self.period).mean()
        rsi = (100 - (100 / (1 + gain / loss))).to_numpy(dtype=float)

        ok = ~np.isnan(rsi) & (rsi >= 0) & (rsi <= 100)
        buy = ok & (rsi < 30)
        sell = ok & (rsi > 70)
        signal = np.where(buy, 1, np.where(sell, -1, 0))
        conf = np.where(buy, (30 - rsi) / 30, (rsi - 70) / 30)

        valid = np.arange(len(rsi)) + 1 >= self.period + 1
        return self._series_frame(data.index, signal, conf, valid)
