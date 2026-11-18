"""Vectorised `generate_signal_series` must equal per-prefix `generate_signal`."""
import numpy as np
import pytest

from config import build_agents

SYMBOLS = ["KGH", "CDR", "PKO", "LPP"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_series_matches_loop(paper_cfg, paper_data, symbol):
    df = paper_data[symbol]
    for agent in build_agents(paper_cfg):
        fast = agent.generate_signal_series(df)
        slow_sig = np.empty(len(df), dtype=int)
        slow_conf = np.empty(len(df), dtype=float)
        for i in range(len(df)):
            s = agent.generate_signal(df.iloc[: i + 1], symbol)
            slow_sig[i], slow_conf[i] = s["signal"], s["confidence"]
        assert (fast["signal"].to_numpy() == slow_sig).all(), agent.name
        np.testing.assert_array_equal(fast["confidence"].to_numpy(), slow_conf, err_msg=agent.name)
