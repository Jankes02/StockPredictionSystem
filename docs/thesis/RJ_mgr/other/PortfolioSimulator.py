from typing import Dict, List, Literal, Optional, TypedDict

import pandas as pd

from src.agents.base.Decision import Decision


class Position(TypedDict):
    symbol: str
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    entry_commission: float


class Trade(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL"]
    price: float
    quantity: int
    date: pd.Timestamp
    pnl: Optional[float]


BPS = 1e-4


class PortfolioSimulator:
    def __init__(
        self,
        initial_cash: float = 100_000,
        position_size: float = 0.1,  # How much capital should be invested
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        stamp_duty_bps: float = 0.0,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_size = position_size
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        # One-way tax charged on purchases only (e.g. UK Stamp Duty Reserve Tax).
        # Sells are not charged. Kept separate from commission so the two legs
        # can carry asymmetric rates.
        self.stamp_duty_bps = stamp_duty_bps

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []

    # ---------- MAIN API ----------

    def process_day(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        decisions: List[Decision],
    ) -> None:
        for d in decisions:
            if d["action"] == "SELL":
                self._sell(d["symbol"], prices.get(d["symbol"]), date)

        for d in decisions:
            if d["action"] == "BUY":
                self._buy(d["symbol"], prices.get(d["symbol"]), date)

        self._record_equity(date, prices)

    # ---------- TRADING ----------

    def _buy(self, symbol: str, price: Optional[float], date: pd.Timestamp) -> None:
        if price is None:
            return
        if symbol in self.positions:
            return

        fill_price = price * (1.0 + self.slippage_bps * BPS)
        # Purchases bear commission plus any one-way stamp duty.
        buy_charge_rate = (self.commission_bps + self.stamp_duty_bps) * BPS

        # Size the position against cash capacity inclusive of entry charges
        allocation = self.cash * self.position_size
        max_spend = allocation / (1.0 + buy_charge_rate)
        if max_spend <= 0:
            return

        quantity = int(max_spend // fill_price)
        if quantity <= 0:
            return

        notional = quantity * fill_price
        entry_charge = notional * buy_charge_rate
        total_cost = notional + entry_charge
        if total_cost > self.cash:
            return

        self.cash -= total_cost

        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_date": date,
            "entry_commission": entry_charge,
        }

        self.trades.append({
            "symbol": symbol,
            "action": "BUY",
            "price": fill_price,
            "quantity": quantity,
            "date": date,
            "pnl": None,
        })

    def _sell(self, symbol: str, price: Optional[float], date: pd.Timestamp) -> None:
        if price is None:
            return
        if symbol not in self.positions:
            return

        fill_price = price * (1.0 - self.slippage_bps * BPS)
        commission_rate = self.commission_bps * BPS

        pos = self.positions.pop(symbol)
        quantity = pos["quantity"]
        notional = quantity * fill_price
        commission = notional * commission_rate
        proceeds = notional - commission
        self.cash += proceeds

        pnl = (fill_price - pos["entry_price"]) * quantity - commission - pos["entry_commission"]

        self.trades.append({
            "symbol": symbol,
            "action": "SELL",
            "price": fill_price,
            "quantity": quantity,
            "date": date,
            "pnl": pnl,
        })

    # ---------- EQUITY ----------

    def _record_equity(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
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

    def summary(self) -> dict:
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