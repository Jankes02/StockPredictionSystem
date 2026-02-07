from base.BaseAgent import BaseAgent
from base.AgentSignal import AgentSignal
import pandas as pd


class MACDAgent(BaseAgent):
    name = "macd"
    horizon = "daily"
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

        signal = 0
        if prev_macd < prev_signal and curr_macd > curr_signal:
            signal = 1
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            signal = -1

        if signal == 0:
            return self._neutral_signal(symbol)

        hist = curr_macd - curr_signal

        confidence = min(abs(hist) / close.iloc[-1] * self.confidence_scale, 1)

        return self._pack(symbol, signal, float(confidence))
