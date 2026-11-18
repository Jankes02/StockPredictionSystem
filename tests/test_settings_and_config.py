import pytest

from src.core.config import load_live_config
from src.core.settings import load_alpaca_settings
from tests.conftest import ROOT


@pytest.fixture
def env_file(monkeypatch, tmp_path):
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path / ".env"


def test_missing_keys_raise(env_file):
    env_file.write_text("")
    with pytest.raises(RuntimeError, match="Missing ALPACA_API_KEY"):
        load_alpaca_settings(env_file)


def test_keys_loaded_and_hidden_in_repr(env_file):
    env_file.write_text("ALPACA_API_KEY=abc\nALPACA_SECRET_KEY=xyz\n")
    s = load_alpaca_settings(env_file)
    assert s.paper and s.api_key.get_secret_value() == "abc"
    assert "xyz" not in repr(s)


def test_live_trading_is_refused(env_file):
    env_file.write_text("ALPACA_API_KEY=a\nALPACA_SECRET_KEY=b\nALPACA_PAPER=false\n")
    with pytest.raises(RuntimeError, match="only paper"):
        load_alpaca_settings(env_file)


def test_live_config_loads_and_rejects_typos(tmp_path):
    cfg = load_live_config(ROOT / "config_live.yaml")
    assert "SPY" in cfg.data_symbols() and cfg.data.feed == "sip"
    bad = tmp_path / "c.yaml"
    bad.write_text("universe: {symbols: [AAPL]}\ndata: {feeed: sip}\n")
    with pytest.raises(Exception, match="feeed"):
        load_live_config(bad)
