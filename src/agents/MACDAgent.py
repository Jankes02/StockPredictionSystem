from src.agents.base.BaseAgent import BaseAgent
from src.agents.base.AgentSignal import AgentSignal
import pandas as pd
import numpy as np


class MACDAgent(BaseAgent):
    name = "macd"
    kind = "momentum"

    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 17,
        signal_period: int = 9,
        confidence_scale: float = 5
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.confidence_scale = confidence_scale

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> AgentSignal:
        min_len = self.slow_period + self.signal_period + 2
        if len(data) < min_len:
            return self._neutral_signal(symbol)

        close = data["Close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_signal, curr_signal = signal_line.iloc[-2], signal_line.iloc[-1]

        histogram = macd_line - signal_line

        signal = 0
        if prev_macd < prev_signal and curr_macd > curr_signal:
            signal = 1
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            signal = -1

        if signal == 0:
            return self._neutral_signal(symbol)

        curr_hist = histogram.iloc[-1]
        window = 50
        hist_std = histogram.tail(window).std()

        if hist_std > 0:
            confidence = float(np.tanh(abs(curr_hist) / (2 * hist_std)))
        else:
            confidence = 0.5

        return self._pack(symbol, signal, float(confidence))

    def generate_signal_series(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        n = len(close)
        min_len = self.slow_period + self.signal_period + 2

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = (macd_line - signal_line).to_numpy(dtype=float)

        m, s = macd_line.to_numpy(dtype=float), signal_line.to_numpy(dtype=float)
        pm, ps = np.roll(m, 1), np.roll(s, 1)
        up = (pm < ps) & (m > s)
        down = (pm > ps) & (m < s)
        signal = np.where(up, 1, np.where(down, -1, 0))
        signal[:1] = 0

        # Confidence only matters on crossover rows, which are rare, so compute
        # `histogram.tail(50).std()` exactly as `generate_signal` does, per row.
        window = 50
        hist_series = pd.Series(histogram)
        conf = np.zeros(n)
        for i in np.flatnonzero(signal):
            hist_std = hist_series.iloc[max(0, i - window + 1): i + 1].std()
            if hist_std > 0:
                conf[i] = float(np.tanh(abs(histogram[i]) / (2 * hist_std)))
            else:
                conf[i] = 0.5

        valid = np.arange(n) + 1 >= min_len
        return self._series_frame(data.index, signal, conf, valid)
