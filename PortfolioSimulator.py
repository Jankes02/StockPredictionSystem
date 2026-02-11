import pandas as pd
from typing import Dict, List, Literal


Position = {
    "symbol": str,
    "quantity": int,
    "entry_price": float,
    "entry_date": pd.Timestamp,
}


Trade = {
    "symbol": str,
    "action": Literal["BUY", "SELL"],
    "price": float,
    "quantity": int,
    "date": pd.Timestamp,
    "pnl": float | None,
}


class PortfolioSimulator:
    def __init__(
        self,
        initial_cash: float = 100_000,
        position_size: float = 0.1,  # How much capital should be invested
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_size = position_size

        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []

    # ---------- MAIN API ----------

    def process_day(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        decisions: List[dict],
    ):
        for d in decisions:
            if d["action"] == "SELL":
                self._sell(d["symbol"], prices.get(d["symbol"]), date)

        for d in decisions:
            if d["action"] == "BUY":
                self._buy(d["symbol"], prices.get(d["symbol"]), date)

        self._record_equity(date, prices)

    # ---------- TRADING ----------

    def _buy(self, symbol: str, price: float | None, date):
        if price is None:
            return
        if symbol in self.positions:
            return

        allocation = self.cash * self.position_size
        if allocation <= 0:
            return

        quantity = int(allocation // price)
        if quantity <= 0:
            return

        cost = quantity * price
        self.cash -= cost

        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": price,
            "entry_date": date,
        }

        self.trades.append({
            "symbol": symbol,
            "action": "BUY",
            "price": price,
            "quantity": quantity,
            "date": date,
            "pnl": None,
        })

    def _sell(self, symbol: str, price: float | None, date):
        if price is None:
            return
        if symbol not in self.positions:
            return

        pos = self.positions.pop(symbol)
        quantity = pos["quantity"]
        proceeds = quantity * price
        self.cash += proceeds

        pnl = (price - pos["entry_price"]) * quantity

        self.trades.append({
            "symbol": symbol,
            "action": "SELL",
            "price": price,
            "quantity": quantity,
            "date": date,
            "pnl": pnl,
        })

    # ---------- EQUITY ----------

    def _record_equity(self, date, prices):
        value = self.cash
        for pos in self.positions.values():
            price = prices.get(pos["symbol"])
            if price is not None:
                value += pos["quantity"] * price

        self.equity_curve.append({
            "date": date,
            "equity": value,
            "cash": self.cash,
            "positions": len(self.positions),
        })

    # ---------- METRICS ----------

    def summary(self):
        realized_pnl = sum(
            t["pnl"] for t in self.trades
            if t["pnl"] is not None
        )

        return {
            "initial_cash": self.initial_cash,
            "final_equity": self.equity_curve[-1]["equity"]
            if self.equity_curve else self.initial_cash,
            "realized_pnl": realized_pnl,
            "num_trades": len(self.trades),
        }