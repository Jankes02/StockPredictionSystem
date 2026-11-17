"""
Regenerate the equity-curve figure (Fig. 2) on the held-out test window,
using results produced by `scripts/run_walk_forward.py`.

Writes `docs/ieee_access/figures/janko2.png`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import get_backtest_options, get_data_dir, load_config
from src.utils.benchmark import build_buy_and_hold_wig20_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "ieee_access"
FIGURES_DIR = DOCS_DIR / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    data_dir = get_data_dir(cfg)
    bt = get_backtest_options(cfg)

    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not equity_path.exists():
        raise RuntimeError(
            f"{equity_path} not found; run scripts/run_walk_forward.py first."
        )

    df = pd.read_csv(equity_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    strategy_curve = [{"date": d, "equity": float(v)} for d, v in df["equity"].items()]
    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, data_dir, bt["initial_cash"]
    )
    bh_df = None
    if bh_curve is not None:
        bh_df = pd.DataFrame(bh_curve).set_index("date")

    # Equity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["equity"], label="Multi-agent system", linewidth=2)
    if bh_df is not None:
        ax.plot(bh_df.index, bh_df["equity"], label="Buy-and-hold WIG20",
                linewidth=2, linestyle="--")
    ax.set_title("Equity curves on the held-out test window", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (PLN)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "janko2.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {FIGURES_DIR / 'janko2.png'}")


if __name__ == "__main__":
    main()
