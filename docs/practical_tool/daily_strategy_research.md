# Daily strategy for the practical tool: research log

This file records how the daily strategy in `strategy_daily.yaml` was chosen, what was rejected, and why
the evidence is still limited. All results come from the same engine the live tool uses: Alpaca SIP daily
bars, splits and dividends adjusted, fills at the next session's open, 5 bps slippage per side and no
commission. Unless a row says otherwise, the backtest window runs from 2017-01 to 2026-11.

## Frozen strategy

```
universe   100 least volatile of the 300 most liquid US stocks (monthly, point in time)
           -> top half by SEC quality score (gross profitability, ROA, cash flow, low leverage, low accruals)
sleeves    50% the paper's 5-agent ensemble (aggressive, min_confidence 0), up to 10 positions
           50% 12-1 month momentum / volatility, top 10, monthly rebalance
risk       each sleeve scales gross exposure to 20% target volatility of SPY (20-day), trims when >10% over
```

Reproduce the evidence with `python -m apps.cli backtest` and `python -m apps.cli robustness-daily`.

| Data set | Sharpe | Benchmark Sharpe | Max DD | Benchmark max DD | CAGR | Benchmark CAGR |
|---|---|---|---|---|---|---|
| US large, 2017–2022 | 0.79 | 0.64 (SPY) | −19.5% | −33.8% | 10.5% | 11.1% |
| US large, 2023–2026 | 1.21 | 1.28 (SPY) | −9.9% | −18.8% | 13.1% | 19.8% |
| US mid caps (ranks 301–1000) | 0.65 | 0.54 (MDY) | −21.6% | −42.2% | 7.5% | 9.5% |
| WIG20, 2022-04 to 2026-03\* | 1.19 | 0.75 | −11.8% | −29.2% | 14.6% | 15.1% |
| FTSE 100, 2022-04 to 2026-03\* | 0.87 | 0.70 | −8.2% | −13.4% | 4.4% | 8.0% |

\* No SEC filings exist for these markets, so the quality filter is off there. Universe: the 10 least volatile index members.

Gate, chosen before the final rounds: on every set, Sharpe must be at least 0.75 × the benchmark's and
|max DD| at most 2/3 of the benchmark's. The strategy passes on all five sets. It is meant to be *safer than
the index*. It is not meant to beat the index in a strong bull market.

## Rounds

1. **Paper ensemble, dynamic top-100 liquidity universe, walk-forward (48 configs).**
   - Result: no edge over SPY.
   - The selected configuration made ~3 trades in 4 years, so it was effectively buy-and-hold.
   - The paper's "risk-aware" behaviour had come from 10%-of-cash sizing, i.e. holding a lot of cash.
2. **Plus exit rules and a market-regime filter (96 configs).**
   - Result: no improvement.
   - Losses came from volatile, high-dollar-volume names falling while the market stayed in an uptrend.
3. **Four families (112 configs).**
   - Families: low-volatility universe, SPY timing, momentum, RSI(2) mean reversion.
   - Key finding: the **low-volatility universe** helps every family.
4. **"Safer" grid (50 configs).**
   - The selected configuration was the ensemble on the low-vol universe with an exit when SPY < SMA100. It passed the gate.
   - Its Sharpe peaks sharply at MA = 100: 0.66 at MA 80 and 0.67 at MA 200.
   - Its deflated Sharpe ratio was 0.79 at 350 trials, which is not significant.
5. **Out-of-sample suite: WIG20, FTSE 100, US mid caps (never used for selection).**
   - The low-vol universe generalises to all three.
   - The SMA100 regime filter fails WIG20 and FTSE, so it was overfit to US data.
   - Volatility targeting replaced the regime filter.
   - The **50/50 blend of ensemble and momentum** was the only candidate to pass all 5 sets.
   - 6 of 8 neighbouring parameter settings (target volatility, window, weights) also pass.
6. **Ways to raise return: index sleeve, concentration, liquid momentum, 3 sleeves, vol target 25%.**
   - Every variant that raised return broke the drawdown gate.
   - The baseline stayed the only candidate passing 5/5.
7. **SEC fundamentals.**
   - A quality filter within the low-vol universe gives a modest, consistent gain: keeping the top 50–70% both work, keeping only 30% is too narrow.
   - Value scores hurt.
   - Quality cannot replace low volatility.
8. **Post-earnings drift (SUE from 10-Q/10-K filings).**
   - Signals are too sparse and arrive too late: filings land a median 36 days after quarter end, about 1–2 weeks after the press release.
   - It did not help.

## Caveats

- **Multiple testing.** About 470 configurations were examined, and the US 2023–2026 window was looked at repeatedly.
  The deflated Sharpe ratio of the closely related blend is about 0.7, so the edge is **not statistically proven**.
  The strongest evidence is the out-of-sample markets. Paper trading on new data is the real test.
- **Survivorship.** The asset pool contains currently listed stocks only, and the WIG20/FTSE checks use current
  constituents. Low-volatility stocks are less affected by this than average, but the bias is not zero.
- **Cash earns nothing in the backtests.** Cash averages 10–60% depending on the market. Holding it in T-bills
  would add roughly 0.5–1.5 pp per year from 2023 on.
- **Fundamentals.** Values are used from the session after the SEC acceptance date. Restatements are used only
  once they are filed, and market caps use unadjusted prices. Foreign IFRS filers have no score and are excluded.

## Code map

| What | Where |
|---|---|
| Engine, risk rules, volatility targeting | `src/engine/daily.py` |
| Decision sources (momentum, mean reversion, trend, score rank, PEAD) | `src/strategies/daily/sources.py` |
| Strategy file schema | `src/strategies/daily/config.py` |
| Walk-forward, grids, gate | `src/research/daily_walk_forward.py` |
| Out-of-sample suite | `src/research/robustness.py`, `src/research/external.py` |
| Deflated Sharpe ratio | `src/research/deflated_sharpe.py` |
| SEC EDGAR data, point-in-time values, factors, SUE | `src/fundamentals/` |
