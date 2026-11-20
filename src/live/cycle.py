"""
One run of the daily paper-trading cycle, after the data has been updated.

    clock -> target session -> data freshness -> sync fills of earlier orders
    -> account equity + drawdown halt -> targets (replay) -> checks
    -> idempotency (already submitted for this session?) -> order plan
    -> dry run: log and store | live: submit market-on-open (OPG) orders

Alpaca accepts OPG orders from 19:00 ET on the previous trading day until
09:28 ET; outside that window a live run refuses to submit.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.data.base import MARKET_TZ
from src.execution.broker import Broker
from src.live.daily import (LiveLimits, OrderPlan, TargetPortfolio, check_plan, check_target, client_order_id,
                            compute_target, drawdown_from_peak, plan_orders)

log = logging.getLogger(__name__)

OPG_CUTOFF = pd.Timedelta(minutes=2)          # Alpaca: 09:28 ET
OPG_EARLIEST = pd.Timedelta(hours=14, minutes=30)   # 19:00 ET the evening before a 09:30 open


@dataclass
class CycleResult:
    run_id: str
    mode: str
    status: str                  # planned | submitted | skipped | halted | failed
    session: Optional[pd.Timestamp] = None
    as_of: Optional[pd.Timestamp] = None
    target: Optional[TargetPortfolio] = None
    plan: Optional[OrderPlan] = None
    equity: float = 0.0
    drawdown: float = 0.0
    messages: List[str] = field(default_factory=list)
    submitted: List[str] = field(default_factory=list)


def opg_window_open(now: pd.Timestamp, next_open: pd.Timestamp) -> bool:
    return next_open - OPG_EARLIEST <= now <= next_open - OPG_CUTOFF


def sync_fills(broker: Broker, db, prefix: str) -> int:
    """Copy the status/fills of our earlier orders from the broker into the state DB."""
    known = db.live_orders()
    if known.empty:
        return 0
    open_ids = set(known.loc[~known["status"].isin(["filled", "canceled", "expired", "rejected"]),
                             "client_order_id"])
    if not open_ids:
        return 0
    after = pd.Timestamp(known["updated_at"].min()) - pd.Timedelta(days=1)
    n = 0
    for o in broker.orders(after=after):
        if o.client_order_id in open_ids:
            db.update_order_fill(o.client_order_id, o.status, o.filled_qty, o.filled_avg_price)
            n += 1
    return n


def run_cycle(*, broker: Broker, db, research, strat, sessions: pd.DatetimeIndex, limits: LiveLimits,
              shadow_start, live: bool, prefix: str = "d1", cash_buffer: float = 0.01,
              min_notional: float = 100.0, replace: bool = False, now: Optional[pd.Timestamp] = None) -> CycleResult:
    """
    `sessions`: exchange trading days (naive dates), e.g. SessionCalendar.sessions.index.
    `research`: DailyResearch built on freshly updated data.
    """
    clock = broker.clock()
    now = pd.Timestamp(now) if now is not None else clock.now
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    mode = "live" if live else "dry"
    res = CycleResult(run_id, mode, "failed")

    def finish(status: str, msg: str = "") -> CycleResult:
        res.status = status
        if msg:
            res.messages.append(msg)
            (log.warning if status in ("failed", "halted") else log.info)("%s", msg)
        db.record_run(run_id, res.session if res.session is not None else now.tz_localize(None),
                      res.as_of if res.as_of is not None else now.tz_localize(None), mode, status,
                      equity=res.equity or None, target_gross=res.target.gross if res.target else None,
                      n_orders=len(res.plan.orders) if res.plan else None, message=" | ".join(res.messages))
        return res

    # 1. which session do the orders target, and is the data up to the close before it?
    next_open = clock.next_open.tz_convert(MARKET_TZ)
    session = next_open.tz_localize(None).normalize()
    res.session = session
    prev = sessions[sessions < session]
    if len(prev) == 0:
        return finish("failed", "session calendar does not cover the previous session")
    expected_as_of = prev[-1]
    last_data = research.calendar[-1]
    if last_data < expected_as_of:
        msg = f"data ends {last_data.date()}, expected the close of {expected_as_of.date()}: run update-data first"
        if live:
            return finish("failed", msg)
        # a dry run may still show what the strategy would do on the last close it has
        res.messages.append(f"{msg} (dry run planned on the close of {last_data.date()} instead)")
        log.warning("%s", res.messages[-1])
        expected_as_of = last_data
    res.as_of = expected_as_of
    if live and not opg_window_open(now.tz_convert(MARKET_TZ), next_open):
        # too early (session in progress, holiday, weekend) is routine for a scheduled job; too late is not
        too_early = now < next_open - OPG_EARLIEST
        return finish("skipped" if too_early else "failed",
                      f"outside the market-on-open order window for {session.date()} "
                      f"(19:00 ET the evening before .. 09:28 ET); nothing submitted")

    # 2. fills of earlier orders, account, drawdown halt
    n = sync_fills(broker, db, prefix)
    if n:
        log.info("updated %d earlier orders from the broker", n)
    acct = broker.account()
    res.equity = acct.equity
    if acct.trading_blocked or acct.status.upper() != "ACTIVE":
        return finish("failed", f"account not tradable (status {acct.status}, blocked={acct.trading_blocked})")
    hist = db.equity_history()
    res.drawdown = drawdown_from_peak(hist["equity"].tolist() if not hist.empty else [], acct.equity)
    db.record_equity(expected_as_of, acct.equity, acct.cash)
    if res.drawdown <= -limits.max_drawdown_halt:
        return finish("halted", f"account drawdown {res.drawdown:.1%} beyond the {limits.max_drawdown_halt:.0%} "
                                f"halt: no orders until reviewed")

    # 3. targets from the replay of the frozen strategy
    target = compute_target(research, strat, shadow_start, expected_as_of)
    res.target = target
    db.record_targets(run_id, target.weights, target.prices)
    problems = check_target(target, limits)
    if problems:
        return finish("failed", "target rejected: " + "; ".join(problems))

    # 4. idempotency and foreign open orders
    tag = f"{prefix}-{session:%Y%m%d}-"
    recent = broker.orders(after=now - pd.Timedelta(days=5))
    ours_today = [o for o in recent if o.client_order_id.startswith(tag)
                  and o.status not in ("canceled", "expired", "rejected")]
    other_open = [o for o in recent if not o.client_order_id.startswith(tag)
                  and o.status in ("new", "accepted", "pending_new", "partially_filled", "held")]
    if other_open:
        res.messages.append("open orders not from this run: " + ", ".join(f"{o.symbol} {o.side} {o.qty:g}"
                                                                          for o in other_open[:10]))
        if live:
            return finish("failed", "cancel or let the other open orders finish before a live run")
    if ours_today and not replace:
        res.plan = OrderPlan([], acct.equity, target.gross)
        return finish("skipped", f"{len(ours_today)} orders for {session.date()} already at the broker "
                                 f"(use --replace to cancel and re-plan)")
    if ours_today and replace and live:
        for o in ours_today:
            broker.cancel(o.broker_id)
        log.info("cancelled %d earlier orders for %s", len(ours_today), session.date())

    # 5. orders
    plan = plan_orders(target, acct.equity, broker.positions(), cash_buffer=cash_buffer, min_notional=min_notional)
    res.plan = plan
    problems = check_plan(plan, limits)
    if problems:
        return finish("failed", "order plan rejected: " + "; ".join(problems))
    for s in plan.skipped:
        res.messages.append(s)

    if not live:
        for o in plan.orders:
            db.record_order(run_id, session, client_order_id(session, o.symbol, o.side, prefix) + "-dry",
                            o.symbol, o.side, o.qty, o.est_price, o.reason, "dry_run")
        return finish("planned", f"dry run: {len(plan.orders)} orders planned for the open of {session.date()}")

    for o in plan.orders:
        cid = client_order_id(session, o.symbol, o.side, prefix)
        try:
            info = broker.submit_market_on_open(o.symbol, o.qty, o.side, cid)
            db.record_order(run_id, session, cid, o.symbol, o.side, o.qty, o.est_price, o.reason, info.status,
                            info.broker_id)
            res.submitted.append(cid)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not block the rest
            db.record_order(run_id, session, cid, o.symbol, o.side, o.qty, o.est_price, o.reason, "rejected")
            res.messages.append(f"{o.symbol} {o.side} {o.qty}: {type(exc).__name__}: {exc}")
            log.error("order %s failed: %s", cid, exc)
    status = "submitted" if len(res.submitted) == len(plan.orders) else "failed"
    return finish(status, f"submitted {len(res.submitted)}/{len(plan.orders)} orders for the open of {session.date()}")
