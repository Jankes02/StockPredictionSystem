"""
Broker interface for the live tool, an Alpaca paper implementation and an
in-memory fake for tests. Only what the daily cycle needs: account, positions,
orders, the market clock, market-on-open orders.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.core.settings import AlpacaSettings

log = logging.getLogger(__name__)


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    status: str = "ACTIVE"
    trading_blocked: bool = False


@dataclass
class OrderInfo:
    client_order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    broker_id: str = ""
    submitted_at: Optional[pd.Timestamp] = None


@dataclass
class Clock:
    now: pd.Timestamp
    is_open: bool
    next_open: pd.Timestamp
    next_close: pd.Timestamp


class Broker:
    def account(self) -> Account:
        raise NotImplementedError

    def positions(self) -> Dict[str, float]:
        """symbol -> quantity (negative = short)."""
        raise NotImplementedError

    def orders(self, after: Optional[pd.Timestamp] = None, status: str = "all") -> List[OrderInfo]:
        raise NotImplementedError

    def clock(self) -> Clock:
        raise NotImplementedError

    def submit_market_on_open(self, symbol: str, qty: int, side: str, client_order_id: str) -> OrderInfo:
        raise NotImplementedError

    def cancel(self, broker_id: str) -> None:
        raise NotImplementedError


def _ts(v) -> Optional[pd.Timestamp]:
    return None if v is None else pd.Timestamp(v)


class AlpacaBroker(Broker):
    """Alpaca Trading API, paper endpoint only (AlpacaSettings refuses live keys)."""

    def __init__(self, settings: AlpacaSettings, client=None):
        if not settings.paper:
            raise RuntimeError("only paper trading is supported")
        if client is None:
            from alpaca.trading.client import TradingClient
            client = TradingClient(settings.api_key.get_secret_value(), settings.secret_key.get_secret_value(),
                                   paper=True)
        self.client = client

    def account(self) -> Account:
        a = self.client.get_account()
        return Account(float(a.equity), float(a.cash), float(a.buying_power), str(getattr(a.status, "value", a.status)),
                       bool(a.trading_blocked or a.account_blocked))

    def positions(self) -> Dict[str, float]:
        # Alpaca reports short positions with a negative qty; the strategy never shorts
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def orders(self, after=None, status="all") -> List[OrderInfo]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus(status), limit=500,
                               after=pd.Timestamp(after).to_pydatetime() if after is not None else None)
        return [self._order(o) for o in self.client.get_orders(filter=req)]

    @staticmethod
    def _order(o) -> OrderInfo:
        return OrderInfo(
            client_order_id=o.client_order_id, symbol=o.symbol, side=str(getattr(o.side, "value", o.side)),
            qty=float(o.qty or 0), status=str(getattr(o.status, "value", o.status)),
            filled_qty=float(o.filled_qty or 0),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price is not None else None,
            broker_id=str(o.id), submitted_at=_ts(o.submitted_at))

    def clock(self) -> Clock:
        c = self.client.get_clock()
        return Clock(pd.Timestamp(c.timestamp), bool(c.is_open), pd.Timestamp(c.next_open), pd.Timestamp(c.next_close))

    def submit_market_on_open(self, symbol, qty, side, client_order_id) -> OrderInfo:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide(side), time_in_force=TimeInForce.OPG,
                                 client_order_id=client_order_id)
        return self._order(self.client.submit_order(order_data=req))

    def cancel(self, broker_id) -> None:
        self.client.cancel_order_by_id(broker_id)


@dataclass
class FakeBroker(Broker):
    """In-memory broker for tests and offline dry runs."""
    cash: float = 100_000.0
    holdings: Dict[str, float] = field(default_factory=dict)
    marks: Dict[str, float] = field(default_factory=dict)
    submitted: List[OrderInfo] = field(default_factory=list)
    now: pd.Timestamp = pd.Timestamp("2026-01-05 08:30", tz="America/New_York")
    next_open: Optional[pd.Timestamp] = None

    def account(self) -> Account:
        eq = self.cash + sum(q * self.marks.get(s, 0.0) for s, q in self.holdings.items())
        return Account(eq, self.cash, self.cash)

    def positions(self):
        return {s: q for s, q in self.holdings.items() if q}

    def orders(self, after=None, status="all"):
        return [o for o in self.submitted if after is None or o.submitted_at is None or o.submitted_at >= after]

    def clock(self):
        nxt = self.next_open or self.now.normalize() + pd.Timedelta(hours=9, minutes=30)
        return Clock(self.now, False, nxt, nxt + pd.Timedelta(hours=6, minutes=30))

    def submit_market_on_open(self, symbol, qty, side, client_order_id):
        if any(o.client_order_id == client_order_id for o in self.submitted):
            raise RuntimeError(f"duplicate client_order_id {client_order_id}")
        o = OrderInfo(client_order_id, symbol, side, float(qty), "accepted", submitted_at=self.now,
                      broker_id=f"fake-{len(self.submitted)}")
        self.submitted.append(o)
        return o

    def cancel(self, broker_id):
        for o in self.submitted:
            if o.broker_id == broker_id:
                o.status = "canceled"
