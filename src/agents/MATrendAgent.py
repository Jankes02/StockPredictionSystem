from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
import numpy as np
import pandas as pd


class MATrendAgent(BaseAgent):
    name = "ma_trend"
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

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        diff = (close.rolling(self.fast).mean() - close.rolling(self.slow).mean()).to_numpy(dtype=float)
        last = close.to_numpy(dtype=float)

        ok = ~np.isnan(last) & (last > 0)
        signal = np.where(ok, np.where(diff > 0, 1, -1), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            conf = np.abs(diff) / last

        valid = np.arange(len(last)) + 1 >= self.slow
        return self._series_frame(data.index, signal, conf, valid)
