from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_curve: List[dict]) -> None:
    df = pd.DataFrame(equity_curve).set_index("date")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()


def plot_equity_with_drawdown(
    equity_curve: List[dict],
    benchmark_curve: Optional[List[dict]] = None,
) -> None:
    df = pd.DataFrame(equity_curve).set_index("date")

    equity = df["equity"]
    # rolling_max = equity.cummax()
    # drawdown = (equity - rolling_max) / rolling_max

    _, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    ax1.plot(equity.index, equity.values, label="Strategy")
    if benchmark_curve:
        bdf = pd.DataFrame(benchmark_curve).set_index("date")
        ax1.plot(bdf.index, bdf["equity"], label="Buy-and-hold WIG20")
    ax1.set_title("Equity Curve", fontsize=18)
    ax1.set_ylabel("Equity", fontsize=14)
    ax1.set_xlabel("Date", fontsize=14)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.grid(True)

    # ax2.fill_between(drawdown.index, drawdown, 0)
    # ax2.set_title("Drawdown")
    # ax2.set_ylabel("Drawdown")
    # ax2.grid(True)

    plt.tight_layout()
    plt.show()
