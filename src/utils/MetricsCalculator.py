import numpy as np
import pandas as pd


class MetricsCalculator:
    def __init__(self, equity_curve: list, trades: list):
        if equity_curve:
            self.equity = pd.DataFrame(equity_curve).set_index("date")
        else:
            self.equity = pd.DataFrame(columns=["equity", "cash", "positions"])
            self.equity.index.name = "date"
        self.trades = pd.DataFrame(trades)
        if not self.trades.empty and "pnl" not in self.trades.columns:
            self.trades["pnl"] = None

    # ---------- RETURNS ----------

    def total_return(self) -> float:
        if self.equity.empty or "equity" not in self.equity.columns:
            return 0.0
        start = self.equity["equity"].iloc[0]
        end = self.equity["equity"].iloc[-1]
        if start <= 0:
            return 0.0
        return (end - start) / start

    def cagr(self) -> float:
        if self.equity.empty or len(self.equity) < 2 or "equity" not in self.equity.columns:
            return 0.0
        days = (self.equity.index[-1] - self.equity.index[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        start = self.equity["equity"].iloc[0]
        end = self.equity["equity"].iloc[-1]
        if start <= 0:
            return 0.0
        return (end / start) ** (1 / years) - 1

    # ---------- DRAWDOWN ----------

    def max_drawdown(self) -> float:
        if self.equity.empty or "equity" not in self.equity.columns:
            return 0.0
        equity = self.equity["equity"]
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    # ---------- RISK ----------

    def sharpe(self, risk_free_rate: float = 0.0) -> float:
        if self.equity.empty or "equity" not in self.equity.columns:
            return 0.0
        returns = self.equity["equity"].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        excess = returns - risk_free_rate / 252
        return float(np.sqrt(252) * excess.mean() / excess.std())

    def calmar_ratio(self) -> float:
        """Return per unit of drawdown risk (CAGR / |max_drawdown|)."""
        dd = self.max_drawdown()
        if dd >= 0:
            return 0.0
        cagr = self.cagr()
        return float(cagr / abs(dd))

    # ---------- TRADING ----------

    def num_trades(self) -> int:
        try:
            closed = self.trades[self.trades["pnl"].notna()]
        except KeyError:
            return 0
        return len(closed)

    def win_rate(self) -> float:
        try:
            closed = self.trades[self.trades["pnl"].notna()]
        except KeyError:
            return 0.0
        if closed.empty:
            return 0.0
        return float((closed["pnl"] > 0).mean())

    # ---------- SUMMARY ----------

    def summary(self) -> dict:
        """Metrics suitable for reports, plots, and thesis (academic standard)."""
        return {
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "max_drawdown": self.max_drawdown(),
            "sharpe": self.sharpe(),
            "calmar_ratio": self.calmar_ratio(),
            "num_trades": self.num_trades(),
            "win_rate": self.win_rate(),
        }
