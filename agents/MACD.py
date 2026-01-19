import pandas as pd

class MACD:
    def __init__(self, fast_period=8, slow_period=17, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period


    def signal(self, data: pd.DataFrame) -> int:
        if len(data) < self.slow_period + self.signal_period:
            return 0

        close = data['Close']

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Przecięcie MACD z linią sygnału
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return 1 # kupno
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return -1 # sprzedaż
        else:
            return 0