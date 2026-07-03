"""
Load config.yaml and provide helpers to build agents, decision agent, and paths.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.agents.MACDAgent import MACDAgent
from src.agents.RSIAgent import RSIAgent
from src.agents.ROCAgent import ROCAgent
from src.agents.BollingerAgent import BollingerAgent
from src.agents.MATrendAgent import MATrendAgent


AGENT_REGISTRY: Dict[str, type] = {
    "macd": MACDAgent,
    "rsi": RSIAgent,
    "roc": ROCAgent,
    "bollinger": BollingerAgent,
    "matrend": MATrendAgent,
}


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load YAML config and resolve paths. Paths are relative to config file directory (project root)."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError("Config file is empty")

    base_dir = Path(config_path).resolve().parent
    data = cfg.get("data", {})
    cfg["data"] = {
        "daily_dir": (base_dir / data.get("daily_dir", "data/daily")).resolve(),
        "index_filename": data.get("index_filename", "wig20.csv"),
    }
    cfg["_base_dir"] = base_dir
    return cfg


def load_config_from_args() -> dict:
    """
    Load the config, honouring an optional `--config <path>` command-line flag.

    Runner scripts call this so the same pipeline can target a different
    market (e.g. `--config config_ftse.yaml`) without code changes. Unknown
    arguments are ignored, so scripts may add their own flags on top.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return load_config(Path(args.config) if args.config else None)


def get_index_filename(cfg: dict) -> str:
    """Return the filename of the market index used for the buy-and-hold benchmark."""
    return (cfg.get("data") or {}).get("index_filename", "wig20.csv")


def get_results_dir(cfg: dict) -> Path:
    """
    Return the directory where result CSVs are written for this config.

    Defaults to `<project>/results`; a config may redirect it (e.g. to
    `results/ftse`) via the `output.results_dir` key. Relative paths are
    resolved against the project root (the config file's directory).
    """
    base_dir = cfg.get("_base_dir") or Path(__file__).resolve().parent
    rel = (cfg.get("output") or {}).get("results_dir", "results")
    return (Path(base_dir) / rel).resolve()


def get_data_dir(cfg: dict) -> Path:
    """Return the data directory Path for the configured backtest frequency."""
    freq = (cfg.get("backtest") or {}).get("data_frequency", "daily")
    return cfg["data"]["daily_dir"]


def build_agents(cfg: dict) -> List[BaseAgent]:
    """Build list of signal agents from config 'agents' section."""
    agents_cfg = cfg.get("agents") or {}
    agents: List[BaseAgent] = []
    for key, params in agents_cfg.items():
        if key not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent in config: {key}. Known: {list(AGENT_REGISTRY.keys())}")
        cls = AGENT_REGISTRY[key]
        if not isinstance(params, dict):
            params = {}
        agents.append(cls(**params))
    return agents


def build_decision_agent(cfg: dict) -> DecisionAgent:
    """Build DecisionAgent from config 'decision' section."""
    dec = cfg.get("decision") or {}
    mode = dec.get("mode", "balanced")
    min_confidence = dec.get("min_confidence", 0.1)
    max_positions = dec.get("max_positions")
    return DecisionAgent(
        mode=mode,
        min_confidence=min_confidence,
        max_positions=max_positions,
    )


def get_backtest_options(cfg: dict) -> dict:
    """Return backtest options: initial_cash, position_size, commission_bps, slippage_bps, stamp_duty_bps."""
    backtest = cfg.get("backtest") or {}
    portfolio = cfg.get("portfolio") or {}
    costs = cfg.get("costs") or {}
    return {
        "initial_cash": backtest.get("initial_cash", 100_000),
        "position_size": portfolio.get("position_size", 0.1),
        "commission_bps": costs.get("commission_bps", 0.0),
        "slippage_bps": costs.get("slippage_bps", 0.0),
        "stamp_duty_bps": costs.get("stamp_duty_bps", 0.0),
    }


def get_evaluation_options(cfg: dict) -> dict:
    """Return evaluation options: mode, train_end, folds."""
    ev = cfg.get("evaluation") or {}
    return {
        "mode": ev.get("mode", "in_sample"),
        "train_end": ev.get("train_end", None),
        "folds": int(ev.get("folds", 3)),
    }
