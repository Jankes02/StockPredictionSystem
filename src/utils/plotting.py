from typing import List

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


def plot_equity_with_drawdown(equity_curve: List[dict]) -> None:
    df = pd.DataFrame(equity_curve).set_index("date")

    equity = df["equity"]
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(equity)
    ax1.set_title("Equity Curve")
    ax1.grid(True)

    ax2.fill_between(drawdown.index, drawdown, 0)
    ax2.set_title("Drawdown")
    ax2.grid(True)

    plt.show()
