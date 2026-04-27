"""
backtest/reporting.py

Formatting and printing utilities for backtest results.
Extracted from backtest/unified.py to reduce responsibilities.

All functions are pure — they take data, print to stdout, and return None.
"""

from __future__ import annotations

import math
from typing import Any


# ──────────────────────────────────────────
# Strategy summary tables
# ──────────────────────────────────────────


def print_backtest_header(mode: str, n_stocks: int, years: int) -> None:
    """Print the header banner for a backtest run."""
    print(f"\n{'='*105}")
    print(f"  UNIFIED BACKTEST ({mode}) — {n_stocks} stocks, {years} years")
    print(f"  Strategy code: strategies.registry (SAME as live engine)")
    print(f"{'='*105}")


def print_backtest_table(summary: dict, header_width: int = 102) -> None:
    """Print detailed per-strategy backtest results table."""
    print(
        f"  {'Strategy':18s} {'Trades':>6s} {'WR':>6s} {'CAGR':>7s} "
        f"{'Expect':>7s} {'Sharpe':>10s} {'PF':>5s} {'MaxDD':>6s} "
        f"{'AvgWin':>7s} {'AvgLoss':>8s} {'Hold':>5s}"
    )
    print(f"  {'-' * header_width}")

    for name, r in sorted(summary.items(), key=lambda x: x[1].cagr, reverse=True):
        if r.total_trades == 0:
            print(f"  {name:18s}  — no trades —")
            continue

        if r.sharpe_valid:
            sharpe_str = f"{r.sharpe:6.2f}"
        elif r.total_trades > 1:
            sharpe_str = f"{r.sharpe:5.2f}(n<30)"
        else:
            sharpe_str = "   N/A"

        verdict = (
            "🏆" if r.cagr > 10 and r.profit_factor > 1.5 else
            "✅" if r.cagr > 5 and r.profit_factor > 1.2 else
            "🟡" if r.cagr > 0 else "❌"
        )

        print(
            f"  {verdict} {name:16s} {r.total_trades:6d} {r.win_rate:5.1f}% "
            f"{r.cagr:6.1f}% {r.expectancy:6.2f}% {sharpe_str:>10s} "
            f"{r.profit_factor:4.2f} {r.max_drawdown:5.1f}% "
            f"{r.avg_win:6.1f}% {r.avg_loss:7.1f}% {r.avg_hold_days:4.0f}d"
        )

    print(f"{'='*105}")


def print_backtest_footer() -> None:
    """Print explanatory footer after the strategy table."""
    print(
        f"  Note: Sharpe marked (n<30) has insufficient trades "
        f"for statistical validity."
    )
    print(
        f"  Primary metrics: CAGR (compound growth), "
        f"Expectancy (avg return/trade), PF (profit/loss ratio)"
    )


def print_compact_summary(summary: dict) -> None:
    """Print one-line per-strategy summary."""
    print()
    for name, r in sorted(summary.items(), key=lambda x: x[1].cagr, reverse=True):
        if r.total_trades == 0:
            continue
        sharpe_str = f"{r.sharpe:.2f}" if r.sharpe_valid else "N/A"
        print(
            f"  {name}: CAGR={r.cagr:.1f}% | "
            f"MaxDD={r.max_drawdown:.1f}% | Sharpe={sharpe_str}"
        )


# ──────────────────────────────────────────
# Benchmark printing
# ──────────────────────────────────────────


def print_benchmarks(
    b_nifty: Any,
    b_eqw: Any,
    b_bnh: Any,
    header_width: int = 105,
) -> None:
    """Print benchmark comparison section."""
    print(f"\n{'='*header_width}")
    print(f"  BENCHMARKS")
    print(f"{'='*header_width}")

    if b_nifty.equity_curve:
        print(
            f"  NIFTY50 Proxy: CAGR={b_nifty.cagr:.1f}% | "
            f"MaxDD={b_nifty.max_drawdown:.1f}%"
        )
    else:
        print(f"  NIFTY50 Proxy: insufficient data")

    if b_eqw.equity_curve:
        print(
            f"  Equal Weight:  CAGR={b_eqw.cagr:.1f}% | "
            f"MaxDD={b_eqw.max_drawdown:.1f}%"
        )
    else:
        print(f"  Equal Weight:  insufficient data")

    if b_bnh.equity_curve:
        print(
            f"  Buy & Hold:    CAGR={b_bnh.cagr:.1f}% | "
            f"MaxDD={b_bnh.max_drawdown:.1f}%"
        )
    else:
        print(f"  Buy & Hold:    insufficient data")

    print(f"{'='*header_width}")
    print(
        f"  Note: Equal Weight and Buy & Hold are identical "
        f"for static allocation."
    )
    print(
        f"  For a meaningfully different third benchmark, "
        f"consider a trend-filtered variant."
    )


# ──────────────────────────────────────────
# Real NIFTY benchmark (^NSEI) printing
# ──────────────────────────────────────────


def print_benchmark_nifty(
    nifty_cagr: float,
    nifty_maxdd: float,
    summary: dict | None = None,
    header_width: int = 105,
) -> None:
    """Print real NIFTY 50 benchmark with per-strategy alpha.

    Parameters
    ----------
    nifty_cagr : float
        CAGR of the real NIFTY 50 index (^NSEI). May be NaN if unavailable.
    nifty_maxdd : float
        Max drawdown of NIFTY 50. May be NaN if unavailable.
    summary : dict | None
        Per-strategy backtest results keyed by strategy name.
        Each value must have ``.cagr`` and ``.total_trades`` attributes.
    header_width : int
        Width of the separator lines for visual alignment.
    """
    if summary is None:
        summary = {}

    print(f"\n{'=' * header_width}")
    print(f"  BENCHMARK")
    print(f"{'=' * header_width}")
    print(f"{'─' * header_width}")

    if nifty_cagr is None or (isinstance(nifty_cagr, float) and math.isnan(nifty_cagr)):
        print(f"  NIFTY50: data unavailable")
    else:
        print(f"  NIFTY50: CAGR={nifty_cagr:.1f}% | MaxDD={nifty_maxdd:.1f}%")

    print(f"{'─' * header_width}")
    print(f"  ALPHA (vs NIFTY):")

    if not summary:
        print(f"    No strategies to evaluate.")
    elif nifty_cagr is None or (isinstance(nifty_cagr, float) and math.isnan(nifty_cagr)):
        print(f"    No benchmark available for alpha computation.")
    else:
        # Sort by strategy CAGR descending
        sorted_strats = sorted(
            summary.items(),
            key=lambda kv: getattr(kv[1], "cagr", 0),
            reverse=True,
        )
        for name, r in sorted_strats:
            if getattr(r, "total_trades", 0) == 0:
                continue
            scagr = getattr(r, "cagr", 0)
            alpha = scagr - nifty_cagr
            sign = "+" if alpha >= 0 else ""
            print(f"    {name:22s} {sign}{alpha:.1f}%")

    print(f"  NOTE: Benchmark assumes full capital deployment.")
    print(f"  Strategy returns include cash drag and execution constraints.")
    print(f"{'=' * header_width}")


# ──────────────────────────────────────────
# Walk-forward formatting
# ──────────────────────────────────────────


def print_walkforward_header(
    n_stocks: int,
    n_windows: int,
    train_years: int,
    test_years: int,
    step_years: int,
) -> None:
    """Print header for walk-forward output."""
    print(f"\n{'='*100}")
    print(
        f"  WALK-FORWARD VALIDATION — {n_stocks} stocks, {n_windows} windows"
    )
    print(
        f"  {train_years}yr train / {test_years}yr test / "
        f"{step_years}yr step"
    )
    print(f"{'='*100}")


def print_walkforward_table(window_results: list[dict]) -> None:
    """Print per-window walk-forward results table."""
    header = (
        f"  {'Window':>6s}  {'Train Start':>12s}  {'Train End':>12s}  "
        f"{'Test Start':>12s}  {'Test End':>12s}  "
        f"{'CAGR':>7s}  {'MaxDD':>7s}  {'Trades':>6s}"
    )
    print(header)
    print(f"  {'-' * 96}")

    for w_idx, w in enumerate(window_results):
        ts_str = w["test_start"].strftime("%Y-%m-%d")
        te_str = w["test_end"].strftime("%Y-%m-%d")
        trs_str = w["train_start"].strftime("%Y-%m-%d")
        tre_str = w["train_end"].strftime("%Y-%m-%d")
        print(
            f"  {w_idx+1:6d}  {trs_str:>12s}  {tre_str:>12s}  "
            f"{ts_str:>12s}  {te_str:>12s}  "
            f"{w['cagr']:6.1f}%  {w['max_dd']:6.1f}%  "
            f"{w['total_trades']:6d}"
        )

    print(f"  {'-' * 96}")


def print_walkforward_summary(wf_summary: dict) -> None:
    """Print summary statistics for walk-forward results."""
    wfe_str = (
        str(wf_summary["wfe"])
        if isinstance(wf_summary["wfe"], str)
        else f"{wf_summary['wfe']:.2f}"
    )
    print(f"  SUMMARY")
    print(f"  {'─' * 40}")
    print(f"    Avg OOS CAGR:    {wf_summary['avg_oos_cagr']:.1f}%")
    print(f"    Avg OOS MaxDD:   {wf_summary['avg_oos_maxdd']:.1f}%")
    print(f"    Avg OOS Trades:  {wf_summary['avg_oos_trades']:.1f}")
    print(f"    % Positive:      {wf_summary['pct_positive_windows']:.1f}%")
    print(f"    WFE:             {wfe_str}")
    print(f"{'='*100}")