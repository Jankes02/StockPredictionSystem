from src.agents.base.BaseAgent import BaseAgent
from src.agents.base.AgentSignal import AgentSignal
import numpy as np
import pandas as pd


class BollingerAgent(BaseAgent):
    name = "bollinger"
    kind = "volatility"

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
        lower_val = lower.iloc[-1]
        upper_val = upper.iloc[-1]

        if pd.isna(last) or last <= 0:
            return self._neutral_signal(symbol)
        if last < lower_val:
            return self._pack(symbol, 1, (lower_val - last) / last)
        if last > upper_val:
            return self._pack(symbol, -1, (last - upper_val) / last)

        return self._neutral_signal(symbol)

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = (ma + self.std_mult * std).to_numpy(dtype=float)
        lower = (ma - self.std_mult * std).to_numpy(dtype=float)
        last = close.to_numpy(dtype=float)

        ok = ~np.isnan(last) & (last > 0)
        buy = ok & (last < lower)
        sell = ok & ~buy & (last > upper)
        signal = np.where(buy, 1, np.where(sell, -1, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            conf = np.where(buy, (lower - last) / last, (last - upper) / last)

        valid = np.arange(len(last)) + 1 >= self.window
        return self._series_frame(data.index, signal, conf, valid)
