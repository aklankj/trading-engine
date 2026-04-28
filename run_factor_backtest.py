#!/usr/bin/env python3
"""
run_factor_backtest.py

Quick-start script for the Factor Engine.
Downloads 12 years of data for NIFTY 200 stocks and runs
a cross-sectional factor backtest.

Usage:
    cd ~/trading-engine
    source venv/bin/activate
    python run_factor_backtest.py                    # Full NIFTY 200
    python run_factor_backtest.py --universe nifty50  # Quick test on 50 stocks
    python run_factor_backtest.py --preset pure_momentum --top-n 30
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Factor Engine Quick Backtest")
    parser.add_argument(
        "--preset",
        choices=["momentum_quality", "pure_momentum", "quality_first", "vol_adjusted"],
        default="momentum_quality",
        help="Factor combination preset",
    )
    parser.add_argument(
        "--universe",
        choices=["nifty50", "nifty100", "nifty200"],
        default="nifty200",
    )
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--freq", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--capital", type=float, default=10_00_000)  # 10L
    parser.add_argument("--weighting", choices=["equal", "score_weighted"], default="equal")
    parser.add_argument("--save", type=str, default="data/factor_backtest_results.json")
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"  🏭 FACTOR ENGINE BACKTEST")
    print(f"{'═'*70}")
    print(f"  Preset:     {args.preset}")
    print(f"  Universe:   {args.universe}")
    print(f"  Top N:      {args.top_n}")
    print(f"  Rebalance:  {args.freq}")
    print(f"  Capital:    ₹{args.capital:,.0f}")
    print(f"  Weighting:  {args.weighting}")
    print(f"  Data years: {args.years}")
    print(f"{'═'*70}\n")

    # ─── Step 1: Fetch price data ────────────────────────────
    from factors.universe import get_universe, fetch_universe_prices, load_fundamental_data
    from factors.composite import CompositeFactorEngine, PRESETS
    from factors.backtest import run_factor_backtest

    symbols = get_universe(args.universe)
    print(f"  Universe: {len(symbols)} stocks")

    t0 = time.time()
    price_data = fetch_universe_prices(symbols, years=args.years)
    t1 = time.time()
    print(f"  Download time: {t1-t0:.0f}s")

    if len(price_data) < 30:
        print(f"\n  ❌ Only {len(price_data)} stocks downloaded. Check internet connection.")
        sys.exit(1)

    # ─── Step 2: Load fundamentals (optional) ────────────────
    fund_data = load_fundamental_data()
    if not fund_data:
        print(f"  ⚠️  No fundamental data found. Quality/Value factors will be skipped.")
        print(f"     Run `python -m fundamental.screener_scraper` first for full power.\n")

    # ─── Step 3: Build engine ────────────────────────────────
    factors = PRESETS[args.preset]()

    # If no fundamental data, filter to price-only factors
    if not fund_data:
        factors = [fc for fc in factors if fc.factor.lookback_days > 0]
        if not factors:
            print("  ❌ No usable factors without fundamental data. Use pure_momentum preset.")
            sys.exit(1)
        # Renormalize weights
        total_w = sum(fc.weight for fc in factors)
        for fc in factors:
            fc.weight = fc.weight / total_w if total_w > 0 else 1.0

    engine = CompositeFactorEngine(
        factors=factors,
        top_n=args.top_n,
        weighting=args.weighting,
    )

    # ─── Step 4: Run backtest ────────────────────────────────
    print(f"\n  Running backtest...")
    t2 = time.time()

    result = run_factor_backtest(
        price_data=price_data,
        engine=engine,
        fundamental_data=fund_data if fund_data else None,
        rebalance_freq=args.freq,
        initial_capital=args.capital,
    )

    t3 = time.time()
    print(f"  Backtest time: {t3-t2:.0f}s")

    # ─── Step 5: Print results ───────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  📊 RESULTS: {result.factor_name}")
    print(f"{'═'*70}")
    print(f"  Period:          {result.start_date} → {result.end_date}")
    print(f"  Total Return:    {result.total_return_pct:+.1f}%")
    print(f"  CAGR:            {result.cagr_pct:+.1f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.1f}%")
    print(f"  Avg Turnover:    {result.avg_monthly_turnover:.1f}%/rebalance")
    print(f"  Rebalances:      {result.total_rebalances}")
    print(f"{'─'*70}")
    print(f"  Benchmark CAGR:  {result.benchmark_cagr_pct:+.1f}%")
    print(f"  ★ ALPHA:         {result.alpha_pct:+.1f}%")
    print(f"  Info Ratio:      {result.information_ratio:.2f}")
    print(f"  Hit Rate:        {result.hit_rate_monthly:.0f}% months > benchmark")
    print(f"{'═'*70}")

    if result.yearly_returns:
        print(f"\n  Year-by-Year Returns:")
        print(f"  {'Year':<8} {'Return':>10}")
        print(f"  {'─'*20}")
        for yr in result.yearly_returns:
            marker = "★" if yr["return_pct"] > 0 else "  "
            print(f"  {yr['year']:<8} {yr['return_pct']:>+9.1f}% {marker}")

    # ─── Step 6: Save results ────────────────────────────────
    from dataclasses import asdict
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full results (without huge equity curve for readability)
    result_dict = asdict(result)
    result_dict["equity_curve"] = result_dict["equity_curve"][:5] + ["..."] + result_dict["equity_curve"][-5:]
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"\n  Results saved: {save_path}")

    # Save full equity curve separately
    eq_path = save_path.parent / "factor_equity_curve.json"
    with open(eq_path, "w") as f:
        json.dump(result.equity_curve, f, default=str)
    print(f"  Equity curve: {eq_path}")

    # ─── Step 7: Verdict ─────────────────────────────────────
    print(f"\n{'═'*70}")
    if result.alpha_pct > 3:
        print(f"  ✅ PROMISING — {result.alpha_pct:+.1f}% alpha over benchmark")
        print(f"     Next: Walk-forward validation to confirm this isn't overfit.")
    elif result.alpha_pct > 0:
        print(f"  ⚠️  MARGINAL — {result.alpha_pct:+.1f}% alpha. May be noise.")
        print(f"     Next: Try quality_first preset or increase universe size.")
    else:
        print(f"  ❌ NO ALPHA — {result.alpha_pct:+.1f}%. Factor combination needs work.")
        print(f"     Next: Try different preset or add more factors.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
