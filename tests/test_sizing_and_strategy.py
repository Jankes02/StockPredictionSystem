"""Equity-based sizing, position/exposure limits, buy priority, strategy config."""
import pandas as pd
import pytest

from src.strategies.daily.config import (
    StrategyDailyConfig,
    load_strategy_daily,
    save_strategy_daily,
)
from src.utils.PortfolioSimulator import PortfolioSimulator

D = pd.Timestamp("2024-01-02")


def _buy(sym, score=1.0):
    return {"symbol": sym, "action": "BUY", "score": score, "confidence": 1.0, "contributing_agents": []}


def test_cash_sizing_is_unchanged_paper_behaviour():
    p = PortfolioSimulator(initial_cash=1000, position_size=0.5)
    p.execute(D, {"A": 1.0, "B": 1.0}, [_buy("A"), _buy("B")])
    assert [p.positions[s]["quantity"] for s in "AB"] == [500, 250]  # 50% of cash, then 50% of the rest


def test_equity_sizing_allocates_fraction_of_equity():
    p = PortfolioSimulator(initial_cash=1000, position_size=0.25, sizing="equity")
    p.execute(D, {"A": 1.0, "B": 1.0, "C": 1.0}, [_buy("A"), _buy("B"), _buy("C")])
    assert [p.positions[s]["quantity"] for s in "ABC"] == [250, 250, 250]


def test_max_positions_and_exposure_limits():
    p = PortfolioSimulator(initial_cash=1000, position_size=0.4, sizing="equity", max_positions=2)
    p.execute(D, {s: 1.0 for s in "ABC"}, [_buy("A"), _buy("B"), _buy("C")])
    assert sorted(p.positions) == ["A", "B"]

    q = PortfolioSimulator(initial_cash=1000, position_size=0.4, sizing="equity", max_gross_exposure=0.5)
    q.execute(D, {s: 1.0 for s in "ABC"}, [_buy("A"), _buy("B"), _buy("C")])
    assert [q.positions[s]["quantity"] for s in q.positions] == [400, 100]  # capped at 50% invested


def test_equity_sizing_uses_marked_value_of_positions():
    p = PortfolioSimulator(initial_cash=1000, position_size=0.5, sizing="equity")
    p.execute(D, {"A": 1.0}, [_buy("A")])          # 500 shares
    p.mark(D, {"A": 3.0})                          # equity = 500 cash + 1500
    p.execute(D, {"B": 1.0}, [_buy("B")])          # wants 1000, only 500 cash
    assert p.positions["B"]["quantity"] == 500


def test_unknown_sizing_rejected():
    with pytest.raises(ValueError):
        PortfolioSimulator(sizing="kelly")


def test_strategy_config_roundtrip_and_validation(tmp_path):
    s = StrategyDailyConfig(calibration={"note": "x"})
    path = save_strategy_daily(s, tmp_path / "s.yaml", header="# frozen\n")
    assert path.read_text(encoding="utf-8").startswith("# frozen\n")
    loaded = load_strategy_daily(path)
    assert loaded == s
    assert len(loaded.build_agents()) == 5
    assert loaded.engine_kwargs()["fill"] == "next_open"
    with pytest.raises(Exception):
        StrategyDailyConfig.model_validate({"decision": {"mode": "yolo"}})
    with pytest.raises(Exception):
        StrategyDailyConfig.model_validate({"portfolio": {"max_gross_exposure": 2.0}})  # no leverage
