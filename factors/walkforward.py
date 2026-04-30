#!/usr/bin/env python3
"""
factors/walkforward.py

Walk-forward validation for the Factor Engine.

Reuses backtest/walkforward.py helpers (generate_walkforward_windows,
summarize_walkforward) — same window logic, different backtester.

For each window:
  - Train period: factor engine "sees" data up to train_end
    (no parameters to fit, but ensures lookback is available)
  - Test period: run factor backtest ONLY on test_start → test_end
  - Compare OOS performance across windows

Usage:
    cd ~/trading-engine && source venv/bin/activate
    nohup python -m factors.walkforward > factor_wf.log 2>&1 &
    tail -f factor_wf.log
"""

from __future__ import annotations

import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from backtest.walkforward import generate_walkforward_windows, summarize_walkforward
from factors.composite import CompositeFactorEngine, PRESETS
from factors.backtest import run_factor_backtest
from factors.universe import get_universe, fetch_universe_prices, load_fundamental_data, get_sector_map


def run_factor_walkforward(
    preset: str = "momentum_quality",
    universe: str = "nifty200",
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    top_n: int = 20,
    years: int = 12,
    capital: float = 1_000_000,
) -> dict:
    """
    Run walk-forward validation on the factor engine.

    Returns summary dict with per-window and aggregate metrics.
    """
    print(f"\n{'='*70}")
    print(f"  FACTOR ENGINE — WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"  Preset:      {preset}")
    print(f"  Universe:    {universe}")
    print(f"  Windows:     {train_years}yr train / {test_years}yr test / {step_years}yr step")
    print(f"  Top N:       {top_n}")
    print(f"  Capital:     ₹{capital:,.0f}")
    print(f"{'='*70}\n")

    # ─── Fetch all data once ──────────────────────────────────
    symbols = get_universe(universe)
    print(f"  Fetching {len(symbols)} stocks...")
    t0 = time.time()
    price_data = fetch_universe_prices(symbols, years=years)
    print(f"  Download: {time.time()-t0:.0f}s")

    fund_data = load_fundamental_data()

    if len(price_data) < 30:
        print(f"  ERROR: Only {len(price_data)} stocks. Need 30+.")
        return {"error": "insufficient_data"}

    # ─── Build unified date index ─────────────────────────────
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    data_index = pd.DatetimeIndex(sorted(all_dates))

    # ─── Generate windows ─────────────────────────────────────
    windows = generate_walkforward_windows(
        data_index,
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )

    if not windows:
        print(f"  ERROR: Data too short for {train_years}+{test_years} year windows.")
        return {"error": "insufficient_history"}

    print(f"  Generated {len(windows)} walk-forward windows\n")

    # ─── Build engine ─────────────────────────────────────────
    factors = PRESETS[preset]()
    # Filter to price-only if no fundamentals
    if not fund_data:
        factors = [fc for fc in factors if fc.factor.lookback_days > 0]
        total_w = sum(fc.weight for fc in factors)
        for fc in factors:
            fc.weight = fc.weight / total_w if total_w > 0 else 1.0

    engine = CompositeFactorEngine(
        factors=factors,
        top_n=top_n,
        weighting="equal",
    )

    # ─── Run each window ──────────────────────────────────────
    window_results = []
    is_results = []

    for w_idx, window in enumerate(windows):
        train_start = window["train_start"]
        train_end = window["train_end"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        print(f"  Window [{w_idx+1}/{len(windows)}]: "
              f"train {train_start.strftime('%Y-%m-%d')}→{train_end.strftime('%Y-%m-%d')} | "
              f"test {test_start.strftime('%Y-%m-%d')}→{test_end.strftime('%Y-%m-%d')}")

        # Run OOS (test period only)
        t1 = time.time()
        oos_result = run_factor_backtest(
            price_data=price_data,
            engine=engine,
            fundamental_data=fund_data if fund_data else None,
            start_date=test_start.strftime("%Y-%m-%d"),
            end_date=test_end.strftime("%Y-%m-%d"),
            initial_capital=capital,
        )

        oos_cagr = oos_result.cagr_pct
        oos_maxdd = oos_result.max_drawdown_pct
        oos_alpha = oos_result.alpha_pct
        oos_sharpe = oos_result.sharpe_ratio

        window_results.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "cagr": oos_cagr,
            "max_dd": oos_maxdd,
            "alpha": oos_alpha,
            "sharpe": oos_sharpe,
            "total_trades": oos_result.total_rebalances,
            "benchmark_cagr": oos_result.benchmark_cagr_pct,
        })

        elapsed = time.time() - t1
        alpha_marker = "+" if oos_alpha > 0 else ""
        print(f"    → CAGR={oos_cagr:+.1f}%  Alpha={alpha_marker}{oos_alpha:.1f}%  "
              f"MaxDD={oos_maxdd:.1f}%  Sharpe={oos_sharpe:.2f}  ({elapsed:.0f}s)")

        # Run IS (train period) for WFE
        is_result = run_factor_backtest(
            price_data=price_data,
            engine=engine,
            fundamental_data=fund_data if fund_data else None,
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            initial_capital=capital,
        )
        is_results.append({
            "cagr": is_result.cagr_pct,
            "max_dd": is_result.max_drawdown_pct,
            "total_trades": is_result.total_rebalances,
        })

    # ─── Summarize ────────────────────────────────────────────
    wf_summary = summarize_walkforward(window_results, is_results)

    # Add factor-specific metrics
    alphas = [w["alpha"] for w in window_results]
    sharpes = [w["sharpe"] for w in window_results]
    wf_summary["avg_oos_alpha"] = round(np.mean(alphas), 1) if alphas else 0
    wf_summary["pct_positive_alpha"] = round(
        sum(1 for a in alphas if a > 0) / len(alphas) * 100, 0
    ) if alphas else 0
    wf_summary["avg_oos_sharpe"] = round(np.mean(sharpes), 2) if sharpes else 0
    wf_summary["min_oos_alpha"] = round(min(alphas), 1) if alphas else 0
    wf_summary["max_oos_alpha"] = round(max(alphas), 1) if alphas else 0

    # ─── Print results ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY — {len(windows)} windows")
    print(f"{'='*70}")

    # Per-window table
    print(f"\n  {'Win':>4} {'Test Period':>25} {'CAGR':>8} {'Alpha':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'─'*62}")
    for i, w in enumerate(window_results):
        period = f"{w['test_start'].strftime('%Y-%m')}-{w['test_end'].strftime('%Y-%m')}"
        print(f"  {i+1:4d} {period:>25} {w['cagr']:+7.1f}% {w['alpha']:+7.1f}% "
              f"{w['max_dd']:7.1f}% {w['sharpe']:7.2f}")

    print(f"\n  {'─'*62}")
    print(f"  Avg OOS CAGR:        {wf_summary['avg_oos_cagr']:+.1f}%")
    print(f"  Avg OOS Alpha:       {wf_summary['avg_oos_alpha']:+.1f}%")
    print(f"  Avg OOS MaxDD:       {wf_summary['avg_oos_maxdd']:.1f}%")
    print(f"  Avg OOS Sharpe:      {wf_summary['avg_oos_sharpe']:.2f}")
    print(f"  % Windows Alpha > 0: {wf_summary['pct_positive_alpha']:.0f}%")
    print(f"  % Windows CAGR > 0:  {wf_summary['pct_positive_windows']:.1f}%")
    print(f"  Alpha Range:         {wf_summary['min_oos_alpha']:+.1f}% to {wf_summary['max_oos_alpha']:+.1f}%")

    wfe = wf_summary.get("wfe", "N/A")
    wfe_str = f"{wfe:.2f}" if isinstance(wfe, (int, float)) else wfe
    print(f"  WFE:                 {wfe_str}")

    # ─── Verdict ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    avg_alpha = wf_summary["avg_oos_alpha"]
    pct_pos = wf_summary["pct_positive_alpha"]

    if avg_alpha > 3 and pct_pos >= 60:
        print(f"  ✅ ROBUST ALPHA — {avg_alpha:+.1f}% avg, positive in {pct_pos:.0f}% of windows")
        print(f"     Ready for paper trading with real capital.")
    elif avg_alpha > 0 and pct_pos >= 50:
        print(f"  ⚠️  MARGINAL — {avg_alpha:+.1f}% avg, positive in {pct_pos:.0f}% of windows")
        print(f"     Edge exists but inconsistent. Consider regime filter.")
    else:
        print(f"  ❌ NO ROBUST ALPHA — {avg_alpha:+.1f}% avg, positive in {pct_pos:.0f}% of windows")
        print(f"     Backtest alpha was likely overfit.")
    print(f"{'='*70}\n")

    # ─── Save ─────────────────────────────────────────────────
    output = {
        "date": datetime.now().isoformat(),
        "preset": preset,
        "universe": universe,
        "train_years": train_years,
        "test_years": test_years,
        "step_years": step_years,
        "top_n": top_n,
        "windows": [
            {
                "window": i + 1,
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "cagr": w["cagr"],
                "alpha": w["alpha"],
                "max_dd": w["max_dd"],
                "sharpe": w["sharpe"],
                "benchmark_cagr": w["benchmark_cagr"],
            }
            for i, w in enumerate(window_results)
        ],
        "summary": {k: v for k, v in wf_summary.items()
                    if not isinstance(v, (pd.Timestamp,))},
    }

    out_path = Path("data/factor_walkforward_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")

    return wf_summary


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor Walk-Forward Validation")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="momentum_quality")
    parser.add_argument("--universe", choices=["nifty50", "nifty100", "nifty200"], default="nifty200")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--years", type=int, default=12)
    args = parser.parse_args()

    run_factor_walkforward(
        preset=args.preset,
        universe=args.universe,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        top_n=args.top_n,
        years=args.years,
    )
