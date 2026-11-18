"""
Regression guard: the published walk-forward results (results/*.csv, produced
by scripts/run_walk_forward.py for the paper) must be reproduced exactly by
the current code on the committed Stooq data. Any refactor of agents, the
decision agent, the simulator or the backtest loop has to keep this green.
"""
import numpy as np
import pandas as pd
import pytest

from config import build_agents, get_backtest_options, get_evaluation_options
from src.utils.walk_forward import WalkForwardEvaluator

from tests.conftest import ROOT

SIM_KEYS = ("initial_cash", "position_size", "commission_bps", "slippage_bps", "stamp_duty_bps")


@pytest.fixture(scope="module")
def evaluator(paper_cfg, paper_data):
    bt = get_backtest_options(paper_cfg)
    assert bt["fill"] == "close", "paper protocol fills on the signal-day Close"
    kwargs = {k: bt[k] for k in SIM_KEYS}
    return WalkForwardEvaluator(build_agents(paper_cfg), paper_data, kwargs)


def _assert_fold_matches(fold, expected: pd.Series):
    assert fold.selected_mode == expected["selected_mode"]
    assert fold.selected_min_confidence == pytest.approx(expected["selected_min_confidence"])
    for prefix, metrics in (("train_", fold.train_metrics), ("test_", fold.test_metrics)):
        for key, value in metrics.items():
            np.testing.assert_allclose(value, expected[prefix + key], rtol=1e-12, err_msg=prefix + key)


def test_headline_split(evaluator, paper_cfg):
    expected = pd.read_csv(ROOT / "results" / "walk_forward_headline.csv").iloc[0]
    train_end = pd.Timestamp(get_evaluation_options(paper_cfg)["train_end"])
    _assert_fold_matches(evaluator.evaluate_anchored_split(train_end=train_end), expected)


def test_k_folds(evaluator, paper_cfg):
    expected = pd.read_csv(ROOT / "results" / "walk_forward_kfolds.csv")
    ev = get_evaluation_options(paper_cfg)
    first_test_start = evaluator.first_date_after(pd.Timestamp(ev["train_end"]))
    folds = evaluator.evaluate_k_folds(first_test_start=first_test_start, folds=ev["folds"])
    assert len(folds) == len(expected)
    for fold, (_, row) in zip(folds, expected.iterrows()):
        _assert_fold_matches(fold, row)
