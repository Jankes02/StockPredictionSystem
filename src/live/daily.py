"""
Live (paper) execution of the frozen daily strategy.

The live tool does not keep its own model of the portfolio. Every run:

1. replays the frozen strategy with the backtest engine from `shadow_start` to
   the last completed session (the same code path as `apps.cli backtest`),
   so live decisions are by construction the backtest's decisions;
2. fills the decisions the engine left for the next open at the last closes
   (`project_next_open`) and reads the resulting **target weights**
   (shadow position value / shadow equity, summed over the sleeves);
3. turns the weights into whole-share targets against the broker's real
   equity and orders the difference to the broker's real positions
   (the broker is the source of truth for what is held).

The shadow portfolio and the broker account can drift apart (fills at the real
open, rounding), but the broker is pulled back to the strategy's weights on
every run, so the drift does not accumulate.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.engine.daily import project_next_open

log = logging.getLogger(__name__)


@dataclass
class TargetPortfolio:
    as_of: pd.Timestamp                    # session whose close the decisions used
    weights: Dict[str, float]              # symbol -> fraction of equity, after the next open
    prices: Dict[str, float]               # last close per symbol (sizing estimate)
    shadow_equity: float
    sleeves: Dict[str, Dict[str, float]] = field(default_factory=dict)   # sleeve -> its own weights
    decisions: List[dict] = field(default_factory=list)                  # pending decisions (with sleeve)

    @property
    def gross(self) -> float:
        return sum(self.weights.values())


def compute_target(research, strat, start, end: Optional[pd.Timestamp] = None) -> TargetPortfolio:
    """Target weights for the next session's open from a replay of `strat`."""
    end = pd.Timestamp(end) if end is not None else research.calendar[-1]
    as_of = research.calendar[research.calendar <= end][-1]
    sleeves = research.run_sleeves(strat, start, as_of)
    values: Dict[str, float] = {}
    prices: Dict[str, float] = {}
    per_sleeve: Dict[str, Dict[str, float]] = {}
    decisions: List[dict] = []
    equity = 0.0
    for name, p in sleeves:
        ctx_prices = _closes(research, strat, name, as_of)
        projected = project_next_open(p, ctx_prices, as_of)
        sleeve_equity = projected.cash + sum(pos["quantity"] * ctx_prices.get(s, pos["entry_price"])
                                             for s, pos in projected.positions.items())
        equity += sleeve_equity
        sv = {}
        for s, pos in projected.positions.items():
            px = ctx_prices.get(s, pos["entry_price"])
            values[s] = values.get(s, 0.0) + pos["quantity"] * px
            sv[s] = pos["quantity"] * px
            prices[s] = px
        per_sleeve[name] = {s: v / sleeve_equity for s, v in sv.items()} if sleeve_equity > 0 else {}
        pe = getattr(p, "pending_execution", None)
        for d in (pe.decisions if pe else []):
            decisions.append({"sleeve": name, "symbol": d["symbol"], "action": d["action"],
                              "score": float(d.get("score", 0.0)), "reason": d.get("reason", "signal")})
        if pe and pe.cap is not None:
            decisions.append({"sleeve": name, "symbol": "*", "action": "CAP", "score": float(pe.cap),
                              "reason": "exposure cap (vol target)"})
    weights = {s: v / equity for s, v in sorted(values.items())} if equity > 0 else {}
    return TargetPortfolio(as_of, weights, prices, equity, per_sleeve, decisions)


def _closes(research, strat, sleeve_name, date) -> Dict[str, float]:
    sub = strat
    if getattr(strat, "type", None) == "blend":
        sub = next(sl.strategy for sl in strat.sleeves if sl.strategy.name == sleeve_name)
    return research.context(sub.universe).prices.closes(date)


# ---------------------------------------------------------------- orders

@dataclass
class PlannedOrder:
    symbol: str
    side: str                 # "buy" | "sell"
    qty: int
    est_price: float
    held_qty: float
    target_qty: int
    target_weight: float
    reason: str

    @property
    def notional(self) -> float:
        return self.qty * self.est_price


@dataclass
class OrderPlan:
    orders: List[PlannedOrder]
    equity: float
    target_gross: float
    skipped: List[str] = field(default_factory=list)


def plan_orders(target: TargetPortfolio, equity: float, positions: Dict[str, float],
                quotes: Optional[Dict[str, float]] = None, cash_buffer: float = 0.01,
                min_notional: float = 100.0) -> OrderPlan:
    """
    Whole-share orders moving `positions` (broker, symbol -> qty) to the
    target weights at `equity` x (1 - cash_buffer). Prices: `quotes` if given,
    else the target's last closes. Changes worth less than `min_notional` are
    skipped unless they close or open a position completely; sells come first.
    """
    px = {**target.prices, **(quotes or {})}
    budget = equity * (1.0 - cash_buffer)
    orders, skipped = [], []
    for sym in sorted(set(target.weights) | set(positions)):
        w = target.weights.get(sym, 0.0)
        held = float(positions.get(sym, 0.0))
        price = px.get(sym)
        if price is None or not math.isfinite(price) or price <= 0:
            if held and w == 0.0:
                skipped.append(f"{sym}: no price, cannot size the sale of {held:g} shares")
            elif w > 0:
                skipped.append(f"{sym}: no price for a target weight of {w:.1%}")
            continue
        tgt = int(math.floor(w * budget / price)) if w > 0 else 0
        diff = tgt - held
        if abs(diff) < 1:
            continue
        full = tgt == 0 or held == 0
        if not full and abs(diff) * price < min_notional:
            skipped.append(f"{sym}: change of {diff:+g} shares (${abs(diff) * price:,.0f}) below the minimum")
            continue
        if diff < 0:
            # sell whole shares; a fractional remainder (from a manual trade) is left alone
            qty = int(math.floor(-diff)) if tgt > 0 else int(math.floor(held))
            if qty < 1:
                continue
            reason = "exit" if tgt == 0 else "reduce"
            orders.append(PlannedOrder(sym, "sell", qty, price, held, tgt, w, reason))
        else:
            reason = "new" if held == 0 else "add"
            orders.append(PlannedOrder(sym, "buy", int(diff), price, held, tgt, w, reason))
    orders.sort(key=lambda o: (o.side != "sell", o.symbol))
    return OrderPlan(orders, equity, target.gross, skipped)


# ---------------------------------------------------------------- checks

@dataclass(frozen=True)
class LiveLimits:
    max_position_weight: float = 0.20     # one name across both sleeves
    max_gross: float = 1.0                # no leverage
    max_positions: int = 25
    max_turnover: float = 1.0             # traded notional / equity in one run
    max_drawdown_halt: float = 0.30       # account drawdown from its peak that halts trading


def check_target(target: TargetPortfolio, limits: LiveLimits) -> List[str]:
    """Problems that must stop the run (empty list = ok)."""
    problems = []
    if target.gross > limits.max_gross + 1e-6:
        problems.append(f"target gross exposure {target.gross:.1%} above {limits.max_gross:.0%}")
    heavy = {s: w for s, w in target.weights.items() if w > limits.max_position_weight + 1e-6}
    if heavy:
        problems.append("position weight above limit: " + ", ".join(f"{s} {w:.1%}" for s, w in heavy.items()))
    if len(target.weights) > limits.max_positions:
        problems.append(f"{len(target.weights)} target positions above {limits.max_positions}")
    return problems


def check_plan(plan: OrderPlan, limits: LiveLimits) -> List[str]:
    problems = []
    if plan.equity <= 0:
        problems.append("account equity is not positive")
        return problems
    turnover = sum(o.notional for o in plan.orders) / plan.equity
    if turnover > limits.max_turnover + 1e-6:
        problems.append(f"turnover {turnover:.0%} of equity above {limits.max_turnover:.0%}")
    return problems


def drawdown_from_peak(equity_history: List[float], current: float) -> float:
    peak = max([*equity_history, current]) if equity_history or current else 0.0
    return 0.0 if peak <= 0 else current / peak - 1.0


def client_order_id(session: pd.Timestamp, symbol: str, side: str, prefix: str = "d1") -> str:
    """Deterministic id: re-running the same session cannot place the same order twice."""
    return f"{prefix}-{pd.Timestamp(session):%Y%m%d}-{symbol}-{side}"
