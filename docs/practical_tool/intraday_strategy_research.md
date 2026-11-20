# Intraday strategies for the practical tool: research log

This file records three pre-registered research rounds on intraday strategies (I1–I3) and the results
that decided what to do next. All trades are opened and closed in the same regular session: no position
is held overnight. Data, costs and engine are described first. The rounds and their results follow.

## Data and engine

- **Bars.** Alpaca SIP data, split- and dividend-adjusted, regular session only. Bars are cut to each
  session's real open and close, so early closes such as 13:00 are handled. The calendar comes from Alpaca's
  `/calendar` endpoint (`src/data/calendar.py`).
  - 5-minute bars, 2020-01 to 2026-11, for 360 symbols. These are the point-in-time top-200 stocks by
    liquidity, recomputed monthly, plus SPY, QQQ, IWM and DIA.
  - 1-minute bars for 12 ETFs.
  - 1-minute bars fetched only where needed: for each session, the top-20 ORB candidates of the full
    universe and of each half (`fetch_fine_bars`, `data/cache/orb_1min`).
- **Engine.** `src/engine/intraday.py` takes a list of trades.
  - Gross exposure is capped at 1 and allocated in entry-time order.
  - Slippage is adverse on both sides of every trade.
  - Market orders fill at the next bar's open. Stop orders fill at the stop level, or at the bar's open if
    the bar gapped through it.
  - If an entry and its stop fall inside the same bar, the stop is assumed hit. This is the pessimistic
    reading.
- **Costs.** Stocks: 3 bps per side, stress test 6 bps. ETFs: 3 bps in I1, then 0.5 bps from I2 onwards,
  stress test 1 bp.
  - The 3 bps level was too high for ETFs: the spread on SPY and QQQ is about 0.1–0.3 bps.
- **Protocol.**
  - Time split: train 2020–2022, test 2023-01 to 2026-11.
  - Stock strategies: the variant is selected on alternate symbols (half A) in the train period and reported
    on the other half (B) and on the test period.
  - ETF strategies: the variant is selected on one ETF and reported on other ETFs.
  - Gate: Sharpe ≥ 0.75 in train, test and out of sample at base costs, and above 0 at stress costs.

## Results

Sharpe ratios are at base costs unless a row says otherwise.

| Family (source) | Round | Best train / test / OOS | Gross edge per trade | Verdict |
|---|---|---|---|---|
| Market intraday momentum, SPY (Gao et al. 2018) | I1 | −0.6 / 0.2 at 0 costs; negative on QQQ, IWM, DIA | ≈ 0 bps | no signal |
| Noise-area momentum (Zarattini, Aziz & Barbon 2024) | I1–I2 | see the table below | 2–6 bps | edge only on tech-heavy ETFs |
| ORB on stocks in play, 5 min (Zarattini, Barbon & Aziz 2024) | I1–I2 | all of the 12 variants are negative with the 1-minute replay | −8 to −15 bps | rejected |
| VWAP reversion | I1 | k = 1: 1.06 / −0.37 | – | worked only in 2020–2022 |
| Gap fade (stocks, ETFs) | I1 | ≤ 0.34 | ≈ 0 | rejected |
| The paper's five agents on 5- and 15-minute bars | I1 | −0.5 to −1.5 | negative | rejected |
| ETF ORB, QQQ (Zarattini & Aziz 2023) | I3 | 0.59 / 0.23 (long only, no target) | 1–4 bps | below gate |
| Cross-sectional intraday momentum / reversal (10:00, 10:30, 15:30) | I3 | all negative | ≈ −1 to +2 bps | rejected |
| Intraday periodicity (Heston, Korajczyk & Sadka 2010) | I3 | −5 to −12 | ≈ +0.2 bps (needs about 260 trades a day) | real but far below costs |
| 30-minute ORB on stocks in play (Crabel 1990) | I3 | −0.3 / 0.27 | ≈ 0 | rejected |

Full tables: `state/reports/intraday_round{1,2,3}.csv`. Scripts used for each round are recorded in the
commit history of this file.

### What the 1-minute replay showed

On 5-minute bars, the result of the stocks-in-play ORB depends entirely on how the entry bar is read.

| Stop | Pessimistic reading | Optimistic reading |
|---|---|---|
| 0.10 × ATR, zero costs | Sharpe about −9 | Sharpe about +4.5 |

Replaying the entry bar on 1-minute bars gives almost exactly the pessimistic result. Any ORB backtest on
5-minute bars with a tight stop is therefore unreliable unless its entry bar is resolved on finer data.

### Noise-area momentum, the only candidate

Costs: 0.5 bps per side, or 1 bp at stress. The parameters are the published ones: a 14-day noise band,
decisions on the half hour, a VWAP/band trailing stop and 2% daily vol targeting with a 1× cap. None of them
was tuned here.

| ETF | Train 2020–22 | Test 2023–26 | Full period, stress | CAGR | Max DD |
|---|---|---|---|---|---|
| QQQ, both sides | 1.32 | 1.39 | 1.09 | ~9–12% | ~−10% |
| QQQ, long only | 1.25 | 1.14 | 0.99 | ~6% | ~−4% |
| SPY, both sides | 1.04 | 0.45 | 0.40 | ~2–8% | ~−11% |

Out of sample on ETFs not used before (full period, both sides): XLK 1.04 and SMH 0.94. XLF, XLE, XLV, EEM,
GLD and TLT are all at 0.2 or below.

The edge is consistent across periods on QQQ, but limited to tech-heavy indices. XLK and SMH are highly
correlated with QQQ, so they are not independent evidence. The strict gate therefore fails. The candidate
also comes from a search over roughly 100 variants, although its parameters were fixed by the literature.

## Conclusions

1. With realistic costs, none of the well-known stock-level intraday effects survives on liquid US large caps.
2. Intraday momentum at the index level survives only on QQQ-like (tech) ETFs.
3. The only intraday candidate is noise-area momentum on QQQ. Paper trading is the honest out-of-sample test.
