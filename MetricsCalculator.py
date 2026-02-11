import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MetricsCalculator:
    def __init__(self, equity_curve: list, trades: list):
        self.equity = pd.DataFrame(equity_curve).set_index("date")
        self.trades = pd.DataFrame(trades)
        if "pnl" not in self.trades.columns:
            self.trades["pnl"] = None

    # ---------- RETURNS ----------

    def total_return(self):
        start = self.equity["equity"].iloc[0]
        end = self.equity["equity"].iloc[-1]
        return (end - start) / start

    def cagr(self):
        days = (self.equity.index[-1] - self.equity.index[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        start = self.equity["equity"].iloc[0]
        end = self.equity["equity"].iloc[-1]
        return (end / start) ** (1 / years) - 1

    # ---------- DRAWDOWN ----------

    def max_drawdown(self):
        equity = self.equity["equity"]
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()

    # ---------- TRADES ----------

    def trade_stats(self):
        closed = self.trades[self.trades["pnl"].notna()]
        if closed.empty:
            return {}

        return {
            "num_trades": len(closed),
            "win_rate": (closed["pnl"] > 0).mean(),
            "avg_pnl": closed["pnl"].mean(),
            "total_pnl": closed["pnl"].sum(),
        }

    # ---------- RISK ----------

    def sharpe(self, risk_free_rate=0.0):
        returns = self.equity["equity"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        excess = returns - risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / excess.std()

    # ---------- SUMMARY ----------

    def summary(self):
        return {
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "max_drawdown": self.max_drawdown(),
            "sharpe": self.sharpe(),
            **self.trade_stats(),
        }
