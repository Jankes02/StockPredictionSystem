"""
Regenerate the four data figures of the master's thesis as vector PDFs.

Unlike `scripts/regenerate_figures.py` and `scripts/regenerate_extra_figures.py`,
which produce raster PNGs sized for the two-column IEEE Access layout, this
script writes vector PDFs sized for the single-column A4 thesis layout and
typeset in Arial 9 pt, as required by the faculty editorial guidelines.

Figure titles are omitted: in the thesis each figure carries a LaTeX caption,
so an embedded title would duplicate it.

Writes into `docs/thesis/RJ_mgr/img/`:
  - fig_equity_oos.pdf     equity curves, ensemble vs buy-and-hold
  - fig_drawdown_oos.pdf   underwater (drawdown) curves
  - fig_rolling_sharpe.pdf rolling 60-day annualised Sharpe
  - fig_corr_matrix.pdf    pairwise agent-signal correlation heatmap

Inputs are the CSV files already produced by the experiment scripts; nothing
is recomputed here.
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

from config import get_backtest_options, get_data_dir, load_config
from src.utils.benchmark import build_buy_and_hold_wig20_curve

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
IMG_DIR = ROOT / "docs" / "thesis" / "RJ_mgr" / "img"

ROLLING_WINDOW = 60  # trading days
TRADING_DAYS = 252

# Single-column A4 text width is roughly 15 cm (5.9 in) under the thesis layout.
WIDE = (5.9, 3.1)
SQUARE = (5.2, 4.3)


def _apply_style() -> None:
    """Arial 9 pt throughout, as required by the editorial guidelines."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed TrueType rather than Type 3
    })


def _load_curves() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not equity_path.exists():
        raise RuntimeError(
            f"{equity_path} not found; run scripts/run_walk_forward.py first."
        )

    df = pd.read_csv(equity_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    cfg = load_config()
    bt = get_backtest_options(cfg)
    strategy_curve = [{"date": d, "equity": float(v)} for d, v in df["equity"].items()]
    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, get_data_dir(cfg), bt["initial_cash"]
    )
    bh_df = pd.DataFrame(bh_curve).set_index("date") if bh_curve is not None else None
    return df, bh_df


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _rolling_sharpe(equity: pd.Series) -> pd.Series:
    returns = equity.pct_change()
    mean = returns.rolling(ROLLING_WINDOW).mean()
    std = returns.rolling(ROLLING_WINDOW).std()
    return np.sqrt(TRADING_DAYS) * mean / std


def plot_equity(df: pd.DataFrame, bh_df: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(df.index, df["equity"], label="Multi-agent system", linewidth=1.4)
    if bh_df is not None:
        ax.plot(bh_df.index, bh_df["equity"], label="Buy-and-hold WIG20",
                linewidth=1.4, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (PLN)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(IMG_DIR / "fig_equity_oos.pdf")
    plt.close(fig)


def plot_drawdown(df: pd.DataFrame, bh_df: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    dd = _drawdown(df["equity"])
    ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.30, linewidth=0)
    ax.plot(dd.index, dd.values * 100, label="Multi-agent system", linewidth=1.4)
    if bh_df is not None:
        dd_bh = _drawdown(bh_df["equity"])
        ax.plot(dd_bh.index, dd_bh.values * 100, label="Buy-and-hold WIG20",
                linewidth=1.4, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(IMG_DIR / "fig_drawdown_oos.pdf")
    plt.close(fig)


def plot_rolling_sharpe(df: pd.DataFrame, bh_df: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    rs = _rolling_sharpe(df["equity"])
    ax.plot(rs.index, rs.values, label="Multi-agent system", linewidth=1.4)
    if bh_df is not None:
        rs_bh = _rolling_sharpe(bh_df["equity"])
        ax.plot(rs_bh.index, rs_bh.values, label="Buy-and-hold WIG20",
                linewidth=1.4, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling {ROLLING_WINDOW}-day Sharpe")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(IMG_DIR / "fig_rolling_sharpe.pdf")
    plt.close(fig)


def plot_correlation() -> None:
    path = RESULTS_DIR / "agent_signal_correlation.csv"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found; run scripts/regenerate_extra_figures.py first."
        )
    corr = pd.read_csv(path, index_col=0)

    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.savefig(IMG_DIR / "fig_corr_matrix.pdf")
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    df, bh_df = _load_curves()
    plot_equity(df, bh_df)
    plot_drawdown(df, bh_df)
    plot_rolling_sharpe(df, bh_df)
    plot_correlation()

    for name in ("fig_equity_oos", "fig_drawdown_oos",
                 "fig_rolling_sharpe", "fig_corr_matrix"):
        print(f"Wrote {IMG_DIR / (name + '.pdf')}")


if __name__ == "__main__":
    main()
