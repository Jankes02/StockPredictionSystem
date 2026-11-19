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
        sizing: Literal["cash", "equity"] = "cash",
        max_positions: Optional[int] = None,
        max_gross_exposure: float = 1.0,
    ):
        """
        sizing="cash" (paper protocol): each buy allocates `position_size` of
        the *remaining cash*, so allocations shrink geometrically.
        sizing="equity": each buy allocates `position_size` of current equity
        (marked at last known prices), limited by available cash, by
        `max_positions` open positions and by gross exposure
        <= `max_gross_exposure` x equity.
        """
        if sizing not in ("cash", "equity"):
            raise ValueError(f"Unknown sizing: {sizing!r}")
        self.sizing = sizing
        self.max_positions = max_positions
        self.max_gross_exposure = max_gross_exposure
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
        self._last_prices: Dict[str, float] = {}

    # ---------- MAIN API ----------

    def process_day(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        decisions: List[Decision],
    ) -> None:
        """Fill `decisions` at `prices` and mark equity at the same prices."""
        self.execute(date, prices, decisions)
        self.mark(date, prices)

    def execute(
        self,
        date: pd.Timestamp,
        fill_prices: Dict[str, float],
        decisions: List[Decision],
    ) -> None:
        """Fill decisions (sells first, freeing cash, then buys) at `fill_prices`."""
        for d in decisions:
            if d["action"] == "SELL":
                self._sell(d["symbol"], fill_prices.get(d["symbol"]), date)

        for d in decisions:
            if d["action"] == "BUY":
                self._buy(d["symbol"], fill_prices.get(d["symbol"]), date)

    def mark(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
        """Record end-of-day equity at mark `prices`."""
        self._record_equity(date, prices)

    # ---------- TRADING ----------

    def _buy(self, symbol: str, price: Optional[float], date: pd.Timestamp) -> None:
        if price is None:
            return
        if symbol in self.positions:
            return

        if self.max_positions is not None and len(self.positions) >= self.max_positions:
            return

        fill_price = price * (1.0 + self.slippage_bps * BPS)
        # Purchases bear commission plus any one-way stamp duty.
        buy_charge_rate = (self.commission_bps + self.stamp_duty_bps) * BPS

        # Size the position against cash capacity inclusive of entry charges
        if self.sizing == "cash":
            allocation = self.cash * self.position_size
        else:
            invested = self._invested_value()
            equity = self.cash + invested
            room = self.max_gross_exposure * equity - invested
            allocation = min(equity * self.position_size, room, self.cash)
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

    def trim_to_exposure(self, date: pd.Timestamp, prices: Dict[str, float], target_gross: float) -> None:
        """
        Sell the same fraction of every position so that gross exposure (at
        `prices`, last known prices otherwise) is at most `target_gross` x equity.
        Partial sells are recorded as SELL trades with `partial: True`.
        """
        marks = {s: prices.get(s, self._last_prices.get(s, p["entry_price"])) for s, p in self.positions.items()}
        invested = sum(p["quantity"] * marks[s] for s, p in self.positions.items())
        equity = self.cash + invested
        if invested <= 0 or invested <= target_gross * equity:
            return
        keep = max(0.0, target_gross * equity / invested)
        for sym in sorted(self.positions):
            pos = self.positions[sym]
            sell_qty = pos["quantity"] - int(pos["quantity"] * keep)
            if sell_qty <= 0 or sym not in prices:
                continue
            if sell_qty >= pos["quantity"]:
                self._sell(sym, prices[sym], date)
                continue
            fill_price = prices[sym] * (1.0 - self.slippage_bps * BPS)
            notional = sell_qty * fill_price
            commission = notional * self.commission_bps * BPS
            self.cash += notional - commission
            share = sell_qty / pos["quantity"]
            entry_comm = pos["entry_commission"] * share
            pos["entry_commission"] -= entry_comm
            pos["quantity"] -= sell_qty
            self.trades.append({
                "symbol": sym, "action": "SELL", "price": fill_price, "quantity": sell_qty,
                "date": date, "pnl": (fill_price - pos["entry_price"]) * sell_qty - commission - entry_comm,
                "partial": True,
            })

    # ---------- EQUITY ----------

    def _invested_value(self) -> float:
        """Market value of open positions at last known prices (entry price if none yet)."""
        return sum(
            pos["quantity"] * self._last_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )

    def _record_equity(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
        # A held symbol without a bar today (halt, data gap, or a calendar that
        # is not the intersection of all symbols) is marked at its last known
        # price instead of silently dropping out of equity.
        self._last_prices.update(prices)
        value = self.cash
        for pos in self.positions.values():
            price = self._last_prices.get(pos["symbol"])
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