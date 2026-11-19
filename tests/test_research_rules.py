"""Blend combination, 'safer than the index' gate, strategy-file round trip for blends."""
import pandas as pd
import pytest

from src.research.daily_walk_forward import CombinedPortfolio, Gate, SplitResult, describe
from src.strategies.daily.config import (
    BlendConfig,
    Sleeve,
    StrategyDailyConfig,
    load_strategy_daily,
    save_strategy_daily,
)


class _P:
    def __init__(self, equity, trades=()):
        d = pd.bdate_range("2024-01-01", periods=len(equity))
        self.equity_curve = [{"date": x, "equity": e, "cash": e / 2, "positions": 1} for x, e in zip(d, equity)]
        self.trades = list(trades)


def test_combined_portfolio_sums_sleeves():
    t1 = {"symbol": "A", "date": pd.Timestamp("2024-01-02")}
    t2 = {"symbol": "B", "date": pd.Timestamp("2024-01-01")}
    c = CombinedPortfolio.combine([_P([50, 55, 60], [t1]), _P([50, 45, 40], [t2])])
    assert [e["equity"] for e in c.equity_curve] == [100, 100, 100]
    assert [e["positions"] for e in c.equity_curve] == [2, 2, 2]
    assert [t["symbol"] for t in c.trades] == ["B", "A"]


def _split(train, test, bench_train, bench_test):
    return SplitResult(train_start=None, train_end=None, test_start=None, test_end=None,
                       selected=None, table=pd.DataFrame(), test_metrics=test,
                       benchmark_metrics=bench_test, p_value_sharpe=1.0,
                       train_metrics=train, benchmark_train_metrics=bench_train)


def test_gate_requires_both_windows():
    b = {"sharpe": 1.0, "max_drawdown": -0.30}
    good = {"sharpe": 0.8, "max_drawdown": -0.15}
    assert Gate().check(_split(good, good, b, b))["passed"]
    bad_dd_train = {"sharpe": 0.9, "max_drawdown": -0.25}   # 0.25 > 2/3 * 0.30
    r = Gate().check(_split(bad_dd_train, good, b, b))
    assert not r["passed"] and r["train"]["drawdown_ok"] is False and r["test"]["drawdown_ok"]
    low_sharpe_test = {"sharpe": 0.7, "max_drawdown": -0.10}
    assert not Gate().check(_split(good, low_sharpe_test, b, b))["passed"]


def test_blend_roundtrip_and_describe(tmp_path):
    blend = BlendConfig(name="b", sleeves=[
        Sleeve(weight=0.5, strategy=StrategyDailyConfig(type="momentum")),
        Sleeve(weight=0.5, strategy=StrategyDailyConfig(type="trend_timing")),
    ])
    path = save_strategy_daily(blend, tmp_path / "s.yaml")
    loaded = load_strategy_daily(path)
    assert isinstance(loaded, BlendConfig) and loaded == blend
    d = describe(loaded)
    assert d["type"] == "blend" and "momentum" in d["params"] and "trend_timing" in d["params"]
    with pytest.raises(Exception):
        BlendConfig(sleeves=[Sleeve(weight=0.7, strategy=StrategyDailyConfig())])
