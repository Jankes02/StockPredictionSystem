"""
Substitute the placeholder macros in `docs/ieee_access/paper.tex` and the
placeholder cells in the result tables with the actual numbers computed by
the walk-forward, ablation, stats, risk-metrics, regime-analysis, and
cost-sensitivity scripts.

The substitutions use simple string matching inside the LaTeX source, so
each macro has a unique name and each table row has a unique marker label.
After running this script the .tex file is ready to compile.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PAPER_PATH = Path(__file__).resolve().parent.parent / "docs" / "ieee_access" / "paper.tex"


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _fmt(x, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(x, digits: int = 2) -> str:
    """Render a fraction as 'xx.xx\\%'. LaTeX-safe."""
    try:
        return f"{float(x) * 100:.{digits}f}\\%"
    except (TypeError, ValueError):
        return "-"


def _ens(value: str) -> str:
    return "\\ensuremath{" + value + "}"


def _read_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _replace_macro(src: str, name: str, value: str) -> str:
    """Replace a `\\newcommand{\\name}{...}` body with the given value."""
    start_marker = "\\newcommand{\\" + name + "}{"
    idx = src.find(start_marker)
    if idx < 0:
        return src
    brace_start = idx + len(start_marker)
    depth = 1
    i = brace_start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    brace_end = i - 1
    return src[:brace_start] + value + src[brace_end:]


# --------------------------------------------------------------------------
# Buy-and-hold reference metrics (needed for several tables)
# --------------------------------------------------------------------------


def _buy_and_hold_metrics_for_oos() -> Dict[str, float]:
    from config import get_backtest_options, get_data_dir, load_config
    from src.utils.benchmark import build_buy_and_hold_wig20_curve
    from src.utils.MetricsCalculator import MetricsCalculator

    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not equity_path.exists():
        return {}
    df = pd.read_csv(equity_path)
    df["date"] = pd.to_datetime(df["date"])
    strategy_curve = df.to_dict("records")

    cfg = load_config()
    bt = get_backtest_options(cfg)
    bh = build_buy_and_hold_wig20_curve(strategy_curve, get_data_dir(cfg), bt["initial_cash"])
    if bh is None:
        return {}
    return MetricsCalculator(bh, []).summary()


# --------------------------------------------------------------------------
# Headline, bootstrap, ablation (existing + slight refactor)
# --------------------------------------------------------------------------


def _headline_and_stats_macros(
    headline: pd.Series,
    stats: pd.DataFrame,
    ablation: pd.DataFrame,
    bh: Dict[str, float],
) -> Dict[str, str]:
    macros: Dict[str, str] = {
        "TRtrain":  _ens(_fmt(headline["train_total_return"])),
        "TRtest":   _ens(_fmt(headline["test_total_return"])),
        "Shtest":   _ens(_fmt(headline["test_sharpe"])),
        "MDDtest":  _ens(_fmt(headline["test_max_drawdown"])),
        "Calmtest": _ens(_fmt(headline["test_calmar_ratio"])),
        "CAGRtest": _ens(_fmt(headline["test_cagr"])),
        "Ntrades":  _ens(str(int(headline["test_num_trades"]))),
        "WinRate":  _ens(_fmt(headline["test_win_rate"])),
        "BestMode": "\\texttt{" + str(headline["selected_mode"]) + "}",
        "BestMinConf": _ens(_fmt(headline["selected_min_confidence"])),
    }

    if not stats.empty:
        for _, row in stats.iterrows():
            cmp = str(row["comparison"])
            if cmp == "ensemble_total_return_ci":
                macros["TRtestCIlow"] = _ens(_fmt(row["ci_low"]))
                macros["TRtestCIhigh"] = _ens(_fmt(row["ci_high"]))
            elif cmp == "ensemble_sharpe_ci":
                macros["ShtestCIlow"] = _ens(_fmt(row["ci_low"]))
                macros["ShtestCIhigh"] = _ens(_fmt(row["ci_high"]))
            elif cmp == "ensemble_vs_bh_total_return":
                macros["PvsBH"] = _ens(_fmt(row["p_value"], 3))
            elif cmp == "ensemble_vs_bh_sharpe":
                macros["PvsBHsharpe"] = _ens(_fmt(row["p_value"], 3))
            elif cmp.startswith("ensemble_vs_only_") and cmp.endswith("_total_return"):
                macros["PvsBestSingle"] = _ens(_fmt(row["p_value"], 3))
                only_name = cmp[len("ensemble_vs_only_") : -len("_total_return")]
                macros["BestSingleName"] = "\\texttt{only\\_" + only_name + "}"

    if not ablation.empty:
        singles = ablation[ablation["variant"].str.startswith("only_")]
        if not singles.empty:
            best = singles.loc[singles["total_return"].idxmax()]
            macros["BestSingleTR"] = _ens(_fmt(best["total_return"]))
            macros["BestSingleName"] = (
                "\\texttt{" + str(best["variant"]).replace("_", r"\_") + "}"
            )

    if bh:
        macros["BHtest"] = _ens(_fmt(bh["total_return"]))
        macros["BHsharpe"] = _ens(_fmt(bh["sharpe"]))
        macros["MDDbh"] = _ens(_fmt(bh["max_drawdown"]))
        macros["CAGRbh"] = _ens(_fmt(bh["cagr"]))
        macros["Calmbh"] = _ens(_fmt(bh["calmar_ratio"]))

    return macros


# --------------------------------------------------------------------------
# Risk-adjusted metrics (Sortino, downside deviation, Ulcer, Martin)
# and volatility-scaled comparison
# --------------------------------------------------------------------------


def _risk_metrics_macros(rm: pd.DataFrame) -> Dict[str, str]:
    if rm.empty:
        return {}
    by_series = rm.set_index("series")
    m: Dict[str, str] = {}

    if "ensemble_oos" in by_series.index:
        e = by_series.loc["ensemble_oos"]
        m["EnsSortino"]  = _ens(_fmt(e["sortino"]))
        m["EnsDownDev"]  = _ens(_fmt(e["downside_deviation"]))
        m["EnsUlcer"]    = _ens(_fmt(e["ulcer_index"]))
        m["EnsMartin"]   = _ens(_fmt(e["martin_ratio"]))
    if "buy_and_hold_oos" in by_series.index:
        b = by_series.loc["buy_and_hold_oos"]
        m["BHSortino"]  = _ens(_fmt(b["sortino"]))
        m["BHDownDev"]  = _ens(_fmt(b["downside_deviation"]))
        m["BHUlcer"]    = _ens(_fmt(b["ulcer_index"]))
        m["BHMartin"]   = _ens(_fmt(b["martin_ratio"]))
    if "ensemble_oos_vol_scaled" in by_series.index:
        v = by_series.loc["ensemble_oos_vol_scaled"]
        m["VolScaledTR"]      = _ens(_fmt(v["total_return"]))
        m["VolScaledCAGR"]    = _ens(_fmt(v["cagr"]))
        m["VolScaledSharpe"]  = _ens(_fmt(v["sharpe"]))
        m["VolScaledMDD"]     = _ens(_fmt(v["max_drawdown"]))
        m["VolScaledCalmar"]  = _ens(_fmt(v["calmar_ratio"]))
        m["VolScaledSortino"] = _ens(_fmt(v["sortino"]))
        m["VolScaledUlcer"]   = _ens(_fmt(v["ulcer_index"]))
        m["Leverage"]         = _ens(_fmt(v["leverage"]))
        m["StrategyTrainVol"] = _ens(_fmt(v["strategy_vol_annual_train"]))
        m["BHOosVol"]         = _ens(_fmt(v["target_vol_annual_oos"]))
        # Excess of vol-scaled TR over BH TR, in percentage points
        if "buy_and_hold_oos" in by_series.index:
            bh_tr = float(by_series.loc["buy_and_hold_oos", "total_return"])
            excess = (float(v["total_return"]) - bh_tr) * 100.0
            m["VolScaledExcessPP"] = _ens(f"{excess:+.2f}")
    return m


# --------------------------------------------------------------------------
# Regime decomposition
# --------------------------------------------------------------------------


def _regime_macros(regime: pd.DataFrame) -> Dict[str, str]:
    if regime.empty:
        return {}
    m: Dict[str, str] = {}
    # Days and shares are the same across series; take from ensemble.
    shares = {}
    ens = regime[regime["series"] == "ensemble"]
    for _, r in ens.iterrows():
        label = str(r["regime"])
        if label == "all":
            m["OOSDays"] = _ens(str(int(r["days"])))
            continue
        shares[label] = float(r["share"])
        m[f"Reg{label.capitalize()}Days"] = _ens(str(int(r["days"])))
        m[f"Reg{label.capitalize()}Share"] = _pct(r["share"])

    def _row_macros(series_name: str, prefix: str) -> None:
        sub = regime[regime["series"] == series_name]
        for _, r in sub.iterrows():
            label = str(r["regime"])
            cap = label.capitalize() if label != "all" else "All"
            m[f"{prefix}{cap}TR"] = _ens(_fmt(r["total_return"]))
            m[f"{prefix}{cap}Sharpe"] = _ens(_fmt(r["sharpe"]))

    _row_macros("ensemble", "Ens")
    _row_macros("buy_and_hold", "BH")

    # Headline correction-regime defense number in percentage points:
    # how much less the ensemble loses during corrections than buy-and-hold.
    try:
        ens_corr = float(
            regime[(regime["series"] == "ensemble") & (regime["regime"] == "correction")]
            ["total_return"].iloc[0]
        )
        bh_corr = float(
            regime[(regime["series"] == "buy_and_hold") & (regime["regime"] == "correction")]
            ["total_return"].iloc[0]
        )
        m["CorrectionDefencePP"] = _ens(f"{(ens_corr - bh_corr) * 100:+.2f}")
        ens_side = float(
            regime[(regime["series"] == "ensemble") & (regime["regime"] == "sideways")]
            ["total_return"].iloc[0]
        )
        bh_side = float(
            regime[(regime["series"] == "buy_and_hold") & (regime["regime"] == "sideways")]
            ["total_return"].iloc[0]
        )
        m["SidewaysExcessPP"] = _ens(f"{(ens_side - bh_side) * 100:+.2f}")
        ens_bull = float(
            regime[(regime["series"] == "ensemble") & (regime["regime"] == "bull")]
            ["total_return"].iloc[0]
        )
        bh_bull = float(
            regime[(regime["series"] == "buy_and_hold") & (regime["regime"] == "bull")]
            ["total_return"].iloc[0]
        )
        m["BullShortfallPP"] = _ens(f"{(ens_bull - bh_bull) * 100:+.2f}")
    except (IndexError, KeyError):
        pass

    return m


# --------------------------------------------------------------------------
# Cost sensitivity (summary scalars only; the full 7x4 table is filled
# separately via `_fill_cost_sensitivity_table`).
# --------------------------------------------------------------------------


def _cost_summary_macros(cs: pd.DataFrame) -> Dict[str, str]:
    if cs.empty:
        return {}
    grid = cs.dropna(subset=["commission_bps", "slippage_bps"])
    if grid.empty:
        return {}
    m: Dict[str, str] = {}

    def _pick(c: float, s: float) -> pd.Series:
        row = grid[(grid["commission_bps"] == c) & (grid["slippage_bps"] == s)]
        return row.iloc[0] if not row.empty else pd.Series(dtype=float)

    frictionless = _pick(0.0, 0.0)
    baseline = _pick(39.0, 5.0)
    worst = _pick(100.0, 20.0)

    if not frictionless.empty:
        m["CostZeroTR"] = _ens(_fmt(frictionless["total_return"]))
        m["CostZeroSharpe"] = _ens(_fmt(frictionless["sharpe"]))
    if not baseline.empty:
        m["CostBaseTR"] = _ens(_fmt(baseline["total_return"]))
        m["CostBaseSharpe"] = _ens(_fmt(baseline["sharpe"]))
    if not worst.empty:
        m["CostWorstTR"] = _ens(_fmt(worst["total_return"]))
        m["CostWorstSharpe"] = _ens(_fmt(worst["sharpe"]))

    # Minimum Sharpe anywhere in the grid (robustness claim anchor).
    m["CostMinSharpe"] = _ens(_fmt(grid["sharpe"].min()))
    return m


# --------------------------------------------------------------------------
# Agent signal correlations
# --------------------------------------------------------------------------


def _correlation_macros(corr: pd.DataFrame) -> Dict[str, str]:
    if corr.empty:
        return {}
    mat = corr.set_index(corr.columns[0])
    agents = list(mat.index)
    off_diag: List[float] = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i < j:
                off_diag.append(float(mat.loc[a, b]))
    if not off_diag:
        return {}
    import numpy as np

    vals = np.array(off_diag, dtype=float)
    m: Dict[str, str] = {
        "CorrMeanAbs": _ens(_fmt(float(np.mean(np.abs(vals))))),
        "CorrMin": _ens(_fmt(float(vals.min()))),
        "CorrMax": _ens(_fmt(float(vals.max()))),
        "CorrN": _ens(str(len(vals))),
    }
    return m


# --------------------------------------------------------------------------
# Deep-learning (LSTM) baseline macros
# --------------------------------------------------------------------------


def _lstm_macros(results_dir: Path, prefix: str = "") -> Dict[str, str]:
    """
    Build LSTM-baseline macros from ml_baselines.csv and ml_baseline_stats.csv.

    With prefix="" the macros are \\LSTMtest etc. (WIG20); with prefix="Ftse"
    they are \\FtseLSTMtest etc. Missing files yield an empty dict so the
    static placeholder defaults in the .tex preamble are left in place.
    """
    ml = _read_if_exists(results_dir / "ml_baselines.csv")
    st = _read_if_exists(results_dir / "ml_baseline_stats.csv")
    m: Dict[str, str] = {}

    if not ml.empty and "series" in ml.columns:
        by = ml.set_index("series")
        if "lstm_oos" in by.index:
            lstm = by.loc["lstm_oos"]
            m[f"{prefix}LSTMtest"] = _ens(_fmt(lstm["total_return"]))
            m[f"{prefix}LSTMsharpe"] = _ens(_fmt(lstm["sharpe"]))
            m[f"{prefix}LSTMmdd"] = _ens(_fmt(lstm["max_drawdown"]))
            m[f"{prefix}LSTMcalmar"] = _ens(_fmt(lstm["calmar_ratio"]))
            m[f"{prefix}LSTMtrades"] = _ens(str(int(lstm["num_trades"])))

    if not st.empty and "comparison" in st.columns:
        for _, r in st.iterrows():
            comparison = str(r["comparison"])
            statistic = str(r["statistic"])
            if comparison == "ensemble_vs_lstm" and statistic == "total_return":
                m[f"{prefix}PensVsLSTM"] = _ens(_fmt(r["p_value"], 3))
            elif comparison == "ensemble_vs_lstm" and statistic == "sharpe":
                m[f"{prefix}PensVsLSTMsharpe"] = _ens(_fmt(r["p_value"], 3))
            elif comparison == "lstm_vs_buy_and_hold" and statistic == "total_return":
                m[f"{prefix}PLSTMvsBH"] = _ens(_fmt(r["p_value"], 3))
    return m


# --------------------------------------------------------------------------
# FTSE 100 external-validation macros (all prefixed with "Ftse")
# --------------------------------------------------------------------------


def _ftse_macros(ftse_dir: Path) -> Dict[str, str]:
    """Build the \\Ftse... macros from the results/ftse pipeline outputs."""
    headline = _read_if_exists(ftse_dir / "walk_forward_headline.csv")
    risk = _read_if_exists(ftse_dir / "risk_metrics.csv")
    stats = _read_if_exists(ftse_dir / "stats.csv")
    regime = _read_if_exists(ftse_dir / "regime_analysis.csv")
    ablation = _read_if_exists(ftse_dir / "ablation.csv")

    m: Dict[str, str] = {}
    if headline.empty:
        return m

    h = headline.iloc[0]
    m["FtseTRtest"] = _ens(_fmt(h["test_total_return"]))
    m["FtseCAGR"] = _ens(_fmt(h["test_cagr"]))
    m["FtseMDD"] = _ens(_fmt(h["test_max_drawdown"]))
    m["FtseSharpe"] = _ens(_fmt(h["test_sharpe"]))
    m["FtseCalmar"] = _ens(_fmt(h["test_calmar_ratio"]))
    m["FtseNtrades"] = _ens(str(int(h["test_num_trades"])))
    m["FtseWinRate"] = _ens(_fmt(h["test_win_rate"]))

    # Symbol count from the FTSE config, if available.
    try:
        from config import load_config

        cfg = load_config(Path(__file__).resolve().parent.parent / "config_ftse.yaml")
        n_sym = len(cfg.get("symbols") or [])
        if n_sym:
            m["FtseNsymbols"] = _ens(str(n_sym))
    except (ImportError, FileNotFoundError, ValueError):
        pass

    if not stats.empty:
        for _, r in stats.iterrows():
            cmp = str(r["comparison"])
            if cmp == "ensemble_total_return_ci":
                m["FtseTRtestCIlow"] = _ens(_fmt(r["ci_low"]))
                m["FtseTRtestCIhigh"] = _ens(_fmt(r["ci_high"]))
            elif cmp == "ensemble_sharpe_ci":
                m["FtseSharpeCIlow"] = _ens(_fmt(r["ci_low"]))
                m["FtseSharpeCIhigh"] = _ens(_fmt(r["ci_high"]))
            elif cmp == "ensemble_vs_bh_total_return":
                m["FtsePvsBH"] = _ens(_fmt(r["p_value"], 3))
            elif cmp == "ensemble_vs_bh_sharpe":
                m["FtsePvsBHsharpe"] = _ens(_fmt(r["p_value"], 3))

    bh_tr = None
    if not risk.empty and "series" in risk.columns:
        by = risk.set_index("series")
        if "ensemble_oos" in by.index:
            e = by.loc["ensemble_oos"]
            m["FtseEnsSortino"] = _ens(_fmt(e["sortino"]))
            m["FtseEnsDownDev"] = _ens(_fmt(e["downside_deviation"]))
            m["FtseEnsUlcer"] = _ens(_fmt(e["ulcer_index"]))
            m["FtseEnsMartin"] = _ens(_fmt(e["martin_ratio"]))
        if "buy_and_hold_oos" in by.index:
            b = by.loc["buy_and_hold_oos"]
            bh_tr = float(b["total_return"])
            m["FtseBHtest"] = _ens(_fmt(b["total_return"]))
            m["FtseBHcagr"] = _ens(_fmt(b["cagr"]))
            m["FtseBHmdd"] = _ens(_fmt(b["max_drawdown"]))
            m["FtseBHsharpe"] = _ens(_fmt(b["sharpe"]))
            m["FtseBHcalmar"] = _ens(_fmt(b["calmar_ratio"]))
            m["FtseBHSortino"] = _ens(_fmt(b["sortino"]))
            m["FtseBHDownDev"] = _ens(_fmt(b["downside_deviation"]))
            m["FtseBHUlcer"] = _ens(_fmt(b["ulcer_index"]))
            m["FtseBHMartin"] = _ens(_fmt(b["martin_ratio"]))
        if "ensemble_oos_vol_scaled" in by.index:
            v = by.loc["ensemble_oos_vol_scaled"]
            m["FtseVolScaledTR"] = _ens(_fmt(v["total_return"]))
            m["FtseVolScaledSharpe"] = _ens(_fmt(v["sharpe"]))
            m["FtseVolScaledMDD"] = _ens(_fmt(v["max_drawdown"]))
            m["FtseLeverage"] = _ens(_fmt(v["leverage"]))
            if bh_tr is not None:
                excess = (float(v["total_return"]) - bh_tr) * 100.0
                m["FtseVolScaledExcessPP"] = _ens(f"{excess:+.2f}")

    if not regime.empty:
        ens = regime[regime["series"] == "ensemble"].set_index("regime")
        bh = regime[regime["series"] == "buy_and_hold"].set_index("regime")
        if "all" in ens.index:
            m["FtseOOSDays"] = _ens(str(int(ens.loc["all", "days"])))
        for label in ("bull", "correction", "sideways"):
            if label in ens.index:
                cap = label.capitalize()
                m[f"FtseReg{cap}Days"] = _ens(str(int(ens.loc[label, "days"])))
                m[f"FtseEns{cap}TR"] = _ens(_fmt(ens.loc[label, "total_return"]))
            if label in bh.index:
                cap = label.capitalize()
                m[f"FtseBH{cap}TR"] = _ens(_fmt(bh.loc[label, "total_return"]))
        if "bull" in ens.index:
            m["FtseRegBullShare"] = _pct(ens.loc["bull", "share"])

    if not ablation.empty and "variant" in ablation.columns:
        singles = ablation[ablation["variant"].str.startswith("only_")]
        if not singles.empty:
            best = singles.loc[singles["total_return"].idxmax()]
            m["FtseBestSingleTR"] = _ens(_fmt(best["total_return"]))
            m["FtseBestSingleName"] = (
                "\\texttt{" + str(best["variant"]).replace("_", r"\_") + "}"
            )

    m.update(_lstm_macros(ftse_dir, prefix="Ftse"))
    return m


# --------------------------------------------------------------------------
# Table filling helpers for tables with variable-length rows
# --------------------------------------------------------------------------


def _fill_grid_table(src: str, grid: pd.DataFrame) -> str:
    if grid.empty:
        return src
    pivot = grid.pivot(index="min_confidence", columns="mode", values="total_return")
    ordered_cols = [c for c in ("passive", "balanced", "aggressive") if c in pivot.columns]
    pivot = pivot[ordered_cols]
    best_idx = pivot.stack().idxmax()
    best_mc, best_mode = best_idx
    lines: List[str] = []
    for mc, row in pivot.iterrows():
        cells = []
        for col in ordered_cols:
            val = row[col]
            cell = _fmt(val)
            if mc == best_mc and col == best_mode:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"${_fmt(mc)}$ & " + " & ".join(cells) + " \\\\")
    rebuilt = "\n".join(lines)
    begin = (
        "\\begin{tabular}{lccc}\n\\toprule\n$c_{\\min}$ & passive & balanced & "
        "aggressive \\\\\n\\midrule\n"
    )
    start = src.find(begin)
    if start < 0:
        return src
    body_start = start + len(begin)
    body_end = src.find("\\bottomrule", body_start)
    if body_end < 0:
        return src
    return src[:body_start] + rebuilt + "\n\\bottomrule" + src[body_end + len("\\bottomrule"):]


def _fill_ablation_tables(src: str, ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return src

    single_rows = {}
    for _, row in ablation.iterrows():
        v = str(row["variant"])
        if v.startswith("only_"):
            single_rows[v[len("only_"):]] = (row["total_return"], row["sharpe"])
    single_ordered = [
        ("macd", "MACD only"),
        ("rsi", "RSI only"),
        ("roc", "ROC only"),
        ("bollinger", "Bollinger only"),
        ("ma_trend", "MA trend only"),
    ]
    for key, label in single_ordered:
        if key in single_rows:
            tr, sh = single_rows[key]
            src = _replace_placeholder_row(src, label, _fmt(tr), _fmt(sh))

    lo_rows = {}
    for _, row in ablation.iterrows():
        v = str(row["variant"])
        if v.startswith("no_"):
            lo_rows[v[len("no_"):]] = (row["total_return"], row["sharpe"])
    lo_ordered = [
        ("macd", "no MACD"),
        ("rsi", "no RSI"),
        ("roc", "no ROC"),
        ("bollinger", "no Bollinger"),
        ("ma_trend", "no MA trend"),
    ]
    for key, label in lo_ordered:
        if key in lo_rows:
            tr, sh = lo_rows[key]
            src = _replace_placeholder_row(src, label, _fmt(tr), _fmt(sh))
    return src


def _replace_placeholder_row(src: str, row_label: str, tr: str, sh: str) -> str:
    import re

    pattern = re.compile(
        r"^(" + re.escape(row_label) + r")(\s+)&\s*\\TODO\s*&\s*\\TODO\s*\\\\",
        flags=re.MULTILINE,
    )
    replacement = r"\1\2& $" + tr + r"$ & $" + sh + r"$ \\\\"
    return pattern.sub(replacement, src)


def _fill_kfold_table(src: str, kfolds: pd.DataFrame) -> str:
    if kfolds.empty:
        return src
    rows = []
    for _, row in kfolds.iterrows():
        rows.append(
            f"{int(row['fold_index']) + 1} & \\texttt{{{row['selected_mode']}}} & "
            f"${_fmt(row['selected_min_confidence'])}$ & "
            f"${_fmt(row['test_total_return'])}$ \\\\"
        )
    rebuilt = "\n".join(rows)
    begin = (
        "\\begin{tabular}{lccc}\n\\toprule\nFold & Selected mode & $c_{\\min}$ & "
        "Test total return \\\\\n\\midrule\n"
    )
    start = src.find(begin)
    if start < 0:
        return src
    body_start = start + len(begin)
    body_end = src.find("\\bottomrule", body_start)
    if body_end < 0:
        return src
    return src[:body_start] + rebuilt + "\n\\bottomrule" + src[body_end + len("\\bottomrule"):]


def _fill_cost_sensitivity_table(src: str, cs: pd.DataFrame) -> str:
    """
    Fills the cost sensitivity table, which is a pivot on (commission_bps x
    slippage_bps) of Sharpe. The target LaTeX has a unique header:

        \begin{tabular}{lcccc}
        \toprule
        Commission (bps) & 0 bp & 5 bp & 10 bp & 20 bp \\
        \midrule
        ...
        \bottomrule
        \end{tabular}
    """
    if cs.empty:
        return src
    grid = cs.dropna(subset=["commission_bps", "slippage_bps"]).copy()
    if grid.empty:
        return src
    pivot = grid.pivot(
        index="commission_bps", columns="slippage_bps", values="sharpe"
    )
    # Ensure column order matches the header.
    slip_cols = sorted([c for c in pivot.columns if pd.notna(c)])
    pivot = pivot[slip_cols]

    rows: List[str] = []
    for commission, row in pivot.iterrows():
        cells = [_fmt(row[c]) for c in slip_cols]
        rows.append(f"{int(commission)} & " + " & ".join(cells) + " \\\\")
    rebuilt = "\n".join(rows)

    # Build header dynamically from slippage columns for robustness.
    header_cells = " & ".join(f"{int(c)} bp" for c in slip_cols)
    begin = (
        "\\begin{tabular}{lcccc}\n\\toprule\nCommission (bps) & "
        + header_cells
        + " \\\\\n\\midrule\n"
    )
    start = src.find(begin)
    if start < 0:
        return src
    body_start = start + len(begin)
    body_end = src.find("\\bottomrule", body_start)
    if body_end < 0:
        return src
    return src[:body_start] + rebuilt + "\n\\bottomrule" + src[body_end + len("\\bottomrule"):]


def _fill_correlation_table(src: str, corr: pd.DataFrame) -> str:
    """
    Fills the 5x5 agent-signal correlation table. Target LaTeX header:

        \begin{tabular}{lccccc}
        \toprule
        & MACD & RSI & ROC & Bollinger & MA trend \\
        \midrule
        ...
        \bottomrule
    """
    if corr.empty:
        return src
    mat = corr.set_index(corr.columns[0])
    display_names = {
        "macd": "MACD",
        "rsi": "RSI",
        "roc": "ROC",
        "bollinger": "Bollinger",
        "ma_trend": "MA trend",
    }
    agents = [a for a in ("macd", "rsi", "roc", "bollinger", "ma_trend") if a in mat.index]

    rows: List[str] = []
    for i, a in enumerate(agents):
        label = display_names.get(a, a)
        cells: List[str] = []
        for j, b in enumerate(agents):
            if j < i:
                # lower triangle: blank (we show an upper-triangular form)
                cells.append("")
            elif j == i:
                cells.append("\\textbf{1.00}")
            else:
                val = float(mat.loc[a, b])
                cells.append(f"{val:+.2f}")
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    rebuilt = "\n".join(rows)

    header_cells = " & ".join(display_names.get(a, a) for a in agents)
    begin = (
        "\\begin{tabular}{lccccc}\n\\toprule\n & "
        + header_cells
        + " \\\\\n\\midrule\n"
    )
    start = src.find(begin)
    if start < 0:
        return src
    body_start = start + len(begin)
    body_end = src.find("\\bottomrule", body_start)
    if body_end < 0:
        return src
    return src[:body_start] + rebuilt + "\n\\bottomrule" + src[body_end + len("\\bottomrule"):]


def _fill_regime_table(src: str, regime: pd.DataFrame) -> str:
    """
    Fills the regime decomposition table. Target LaTeX header:

        \begin{tabular}{lrrrrrr}
        \toprule
         Regime & Days & Share & Ensemble TR & Ensemble Sharpe & B\&H TR & B\&H Sharpe \\
        \midrule
    """
    if regime.empty:
        return src
    display = {"bull": "Bull", "correction": "Correction", "sideways": "Sideways", "all": "All"}
    ens = regime[regime["series"] == "ensemble"].set_index("regime")
    bh = regime[regime["series"] == "buy_and_hold"].set_index("regime")

    rows: List[str] = []
    for label in ("bull", "correction", "sideways", "all"):
        if label not in ens.index or label not in bh.index:
            continue
        e = ens.loc[label]
        b = bh.loc[label]
        days = int(e["days"])
        share = _pct(e["share"], 1)
        rows.append(
            f"{display[label]} & {days} & {share} & "
            f"${_fmt(e['total_return'])}$ & ${_fmt(e['sharpe'])}$ & "
            f"${_fmt(b['total_return'])}$ & ${_fmt(b['sharpe'])}$ \\\\"
        )
    rebuilt = "\n".join(rows)

    begin = (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Regime & Days & Share & Ensemble TR & Ensemble Sharpe & B\\&H TR & "
        "B\\&H Sharpe \\\\\n\\midrule\n"
    )
    start = src.find(begin)
    if start < 0:
        return src
    body_start = start + len(begin)
    body_end = src.find("\\bottomrule", body_start)
    if body_end < 0:
        return src
    return src[:body_start] + rebuilt + "\n\\bottomrule" + src[body_end + len("\\bottomrule"):]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    src = PAPER_PATH.read_text(encoding="utf-8")

    headline = _read_if_exists(RESULTS_DIR / "walk_forward_headline.csv")
    grid     = _read_if_exists(RESULTS_DIR / "walk_forward_grid_train.csv")
    ablation = _read_if_exists(RESULTS_DIR / "ablation.csv")
    stats    = _read_if_exists(RESULTS_DIR / "stats.csv")
    kfolds   = _read_if_exists(RESULTS_DIR / "walk_forward_kfolds.csv")
    risk_m   = _read_if_exists(RESULTS_DIR / "risk_metrics.csv")
    regime   = _read_if_exists(RESULTS_DIR / "regime_analysis.csv")
    cost     = _read_if_exists(RESULTS_DIR / "cost_sensitivity.csv")
    corr     = _read_if_exists(RESULTS_DIR / "agent_signal_correlation.csv")

    if headline.empty:
        raise RuntimeError("walk_forward_headline.csv is missing or empty")

    h = headline.iloc[0]
    bh = _buy_and_hold_metrics_for_oos()

    macros: Dict[str, str] = {}
    macros.update(_headline_and_stats_macros(h, stats, ablation, bh))
    macros.update(_risk_metrics_macros(risk_m))
    macros.update(_regime_macros(regime))
    macros.update(_cost_summary_macros(cost))
    macros.update(_correlation_macros(corr))
    macros.update(_lstm_macros(RESULTS_DIR, prefix=""))
    macros.update(_ftse_macros(RESULTS_DIR / "ftse"))

    for name, value in macros.items():
        src = _replace_macro(src, name, value)

    src = _fill_grid_table(src, grid)
    src = _fill_ablation_tables(src, ablation)
    src = _fill_kfold_table(src, kfolds)
    src = _fill_cost_sensitivity_table(src, cost)
    src = _fill_correlation_table(src, corr)
    src = _fill_regime_table(src, regime)

    PAPER_PATH.write_text(src, encoding="utf-8")
    print(f"Filled {len(macros)} macros and result tables in {PAPER_PATH}")


if __name__ == "__main__":
    main()
