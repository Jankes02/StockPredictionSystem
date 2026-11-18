"""Fill-timing semantics of backtest_portfolio_daily on a hand-built example."""
import pandas as pd

from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.utils.backtesting import backtest_portfolio_daily


class ScriptedAgent(BaseAgent):
    """Emits BUY on day index 1 and SELL on day index 3."""
    name = "scripted"
    kind = "trend"

    def generate_signal(self, data, symbol):
        i = len(data) - 1
        if i == 1:
            return self._pack(symbol, 1, 1.0)
        if i == 3:
            return self._pack(symbol, -1, 1.0)
        return self._neutral_signal(symbol)


def _data():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return {"AAA": pd.DataFrame({
        "Open":  [10.0, 11.0, 12.0, 13.0, 14.0],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
    }, index=idx)}


def _run(fill):
    return backtest_portfolio_daily(
        agents=[ScriptedAgent()],
        data_by_symbol=_data(),
        decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
        position_size=1.0,
        fill=fill,
    )


def test_close_fill_uses_signal_day_close():
    trades = _run("close").trades
    assert [(t["action"], t["price"], t["date"].day) for t in trades] == [
        ("BUY", 11.5, 2), ("SELL", 13.5, 4)]


def test_next_open_fill_uses_following_open():
    p = _run("next_open")
    assert [(t["action"], t["price"], t["date"].day) for t in p.trades] == [
        ("BUY", 12.0, 3), ("SELL", 14.0, 5)]
    # Equity is still marked at the Close every day.
    assert len(p.equity_curve) == 5
    day3 = p.equity_curve[2]
    assert day3["equity"] == day3["cash"] + p.trades[0]["quantity"] * 12.5
