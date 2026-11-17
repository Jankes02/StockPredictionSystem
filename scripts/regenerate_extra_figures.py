"""
Generate Figs. 3--5 for the paper (underwater drawdown, rolling Sharpe,
agent-signal correlation matrix).

Writes into `docs/ieee_access/figures/`:
  - janko3.png  underwater drawdown overlay (ensemble vs buy-and-hold)
  - janko4.png  rolling 60-day annualised Sharpe
  - janko5.png  pairwise agent-signal correlation heatmap
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    build_agents,
    get_backtest_options,
    get_data_dir,
    load_config,
)
from main import load_price_data
from src.utils.backtesting import _common_dates
from src.utils.benchmark import build_buy_and_hold_wig20_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "ieee_access"
FIGURES_DIR = DOCS_DIR / "figures"
ROLLING_WINDOW = 60  # trading days


def _build_agent_signal_panel(
    cfg: dict,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a (dates x agents) DataFrame of per-day cross-sectional signals.

    For each date d and agent a we compute:
        s_{a,d} = mean over symbols of signal_{a,d,sym} * confidence_{a,d,sym}
    i.e. a signed, confidence-weighted cross-sectional directional exposure
    the agent recommends on that day. Correlation is taken across agents on
    those daily series.
    """
    agents = build_agents(cfg)
    symbols = cfg.get("symbols") or []
    data_by_symbol = load_price_data(get_data_dir(cfg), symbols)
    all_dates = _common_dates(data_by_symbol)
    window_dates = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

    agent_names = [a.name for a in agents]
    panel = pd.DataFrame(
        index=window_dates, columns=agent_names, dtype=float
    )
    for date in window_dates:
        per_agent_vals = {name: [] for name in agent_names}
        for symbol, df in data_by_symbol.items():
            data = df.loc[:date]
            if len(data) < 2:
                continue
            for agent in agents:
                sig = agent.generate_signal(data, symbol)
                per_agent_vals[agent.name].append(
                    float(sig["signal"]) * float(sig["confidence"])
                )
        for name, vals in per_agent_vals.items():
            panel.loc[date, name] = float(np.mean(vals)) if vals else 0.0
    return panel


def _plot_correlation_matrix(panel: pd.DataFrame) -> None:
    corr = panel.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(
        "Pairwise correlation of agent signals (OOS window)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "janko5.png", dpi=150)
    plt.close(fig)

    # Persist the matrix for the paper table / macros if needed.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(RESULTS_DIR / "agent_signal_correlation.csv")


def _equity_df_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    sharpe = (mean / std) * np.sqrt(252)
    return sharpe


def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    bt = get_backtest_options(cfg)
    data_dir = get_data_dir(cfg)

    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not equity_path.exists():
        raise RuntimeError("Run scripts/run_walk_forward.py before this script.")

    strat_df = _equity_df_from_csv(equity_path)
    strategy_curve = [
        {"date": d, "equity": float(v)} for d, v in strat_df["equity"].items()
    ]
    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, data_dir, bt["initial_cash"]
    )
    if bh_curve is None:
        raise RuntimeError("Cannot build WIG20 buy-and-hold curve")
    bh_df = pd.DataFrame(bh_curve).set_index("date")

    test_start = strat_df.index[0]
    test_end = strat_df.index[-1]

    # ---- Figure 1: agent signal correlation matrix -------------------------
    print(f"Building per-agent signal panel over {test_start.date()}..{test_end.date()} ...")
    panel = _build_agent_signal_panel(cfg, test_start, test_end)
    print("Signal panel shape:", panel.shape)
    _plot_correlation_matrix(panel)

    # ---- Figure 2: rolling 60-day Sharpe -----------------------------------
    strat_ret = strat_df["equity"].pct_change().dropna()
    shared_idx = strat_ret.index.intersection(bh_df.index)
    strat_ret = strat_ret.reindex(shared_idx)
    bh_ret = bh_df["equity"].reindex(shared_idx).pct_change().dropna()
    common = strat_ret.index.intersection(bh_ret.index)
    strat_ret = strat_ret.reindex(common)
    bh_ret = bh_ret.reindex(common)

    strat_roll = _rolling_sharpe(strat_ret, ROLLING_WINDOW)
    bh_roll = _rolling_sharpe(bh_ret, ROLLING_WINDOW)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.plot(strat_roll.index, strat_roll.values, label="Multi-agent system", linewidth=1.8)
    ax.plot(
        bh_roll.index,
        bh_roll.values,
        label="Buy-and-hold WIG20",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_title(
        f"Rolling {ROLLING_WINDOW}-day annualized Sharpe (OOS window)", fontsize=13
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "janko4.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: overlaid underwater drawdown ----------------------------
    strat_dd = _drawdown_series(strat_df["equity"])
    bh_dd = _drawdown_series(bh_df["equity"].reindex(strat_df.index, method="ffill"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(
        strat_dd.index,
        strat_dd.values * 100,
        0.0,
        alpha=0.35,
        label="Multi-agent system",
    )
    ax.fill_between(
        bh_dd.index,
        bh_dd.values * 100,
        0.0,
        alpha=0.25,
        label="Buy-and-hold WIG20",
    )
    ax.plot(strat_dd.index, strat_dd.values * 100, linewidth=1.2)
    ax.plot(bh_dd.index, bh_dd.values * 100, linewidth=1.2, linestyle="--")
    ax.set_title("Underwater drawdown (OOS window)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from peak (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "janko3.png", dpi=150)
    plt.close(fig)

    for name in ("janko3.png", "janko4.png", "janko5.png"):
        print(f"Wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
