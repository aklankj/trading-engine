"""
factors/backtest.py

Cross-sectional factor backtest engine.

This is fundamentally different from strategies/backtest:
  - Strategy backtest: per-stock entry/exit signals, trade-level P&L
  - Factor backtest: monthly ranking → rebalance → portfolio-level returns

The backtest simulates:
  1. At each rebalance date (monthly), rank the universe
  2. Select top N stocks
  3. Buy/sell to match target weights
  4. Track portfolio value daily between rebalances
  5. Deduct transaction costs on turnover
  6. Compare against NIFTY 50 buy-and-hold benchmark

This produces the CAGR, Sharpe, alpha, and drawdown numbers
that determine whether the factor engine has real alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict

from factors.base import FactorBacktestResult
from factors.composite import CompositeFactorEngine, FactorConfig
from factors.universe import get_sector_map


def run_factor_backtest(
    price_data: dict[str, pd.DataFrame],
    engine: CompositeFactorEngine,
    fundamental_data: dict[str, dict] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    rebalance_freq: str = "monthly",  # "monthly" or "quarterly"
    initial_capital: float = 1_000_000,
    transaction_cost_pct: float = 0.002,  # 0.2% round-trip
    benchmark_symbol: str | None = None,  # e.g. "NIFTY 50" proxy
) -> FactorBacktestResult:
    """
    Run a full factor backtest with monthly/quarterly rebalancing.

    Args:
        price_data: {symbol: DataFrame} with DatetimeIndex, 'close' column
        engine: Configured CompositeFactorEngine
        fundamental_data: Optional {symbol: dict} from Screener
        start_date: Backtest start (YYYY-MM-DD). Default: 2 years into data
        end_date: Backtest end. Default: latest data
        rebalance_freq: "monthly" or "quarterly"
        initial_capital: Starting capital in INR
        transaction_cost_pct: One-way cost (applied on buys AND sells)
        benchmark_symbol: Symbol for benchmark comparison (None = equal-weight universe)

    Returns:
        FactorBacktestResult with CAGR, Sharpe, alpha, equity curve, etc.
    """
    sector_map = get_sector_map()

    # ─── Step 1: Build a common date index ────────────────────
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    if not all_dates:
        return FactorBacktestResult(factor_name=_engine_name(engine), start_date="", end_date="")

    all_dates = pd.DatetimeIndex(all_dates)

    # Determine start/end
    if start_date:
        bt_start = pd.Timestamp(start_date)
    else:
        # Start 2 years in to ensure enough lookback for momentum
        bt_start = all_dates[0] + timedelta(days=730)

    if end_date:
        bt_end = pd.Timestamp(end_date)
    else:
        bt_end = all_dates[-1]

    # Filter dates to backtest period
    bt_dates = all_dates[(all_dates >= bt_start) & (all_dates <= bt_end)]
    if len(bt_dates) < 60:
        print(f"  Too few trading days ({len(bt_dates)}) for factor backtest")
        return FactorBacktestResult(factor_name=_engine_name(engine), start_date=str(bt_start)[:10], end_date=str(bt_end)[:10])

    # ─── Step 2: Generate rebalance dates ─────────────────────
    rebal_dates = _get_rebalance_dates(bt_dates, rebalance_freq)
    print(f"  Backtest: {bt_dates[0].strftime('%Y-%m-%d')} to {bt_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Rebalance dates: {len(rebal_dates)} ({rebalance_freq})")

    # ─── Step 3: Simulate ─────────────────────────────────────
    cash = initial_capital
    holdings: dict[str, int] = {}         # {symbol: shares}
    entry_prices: dict[str, float] = {}   # For cost tracking

    equity_curve = []
    monthly_returns = []
    prev_equity = initial_capital
    total_turnover = 0.0
    rebalance_count = 0

    # Build daily close price matrix for fast lookup
    close_matrix = _build_close_matrix(price_data, bt_dates)

    for i, date in enumerate(bt_dates):
        # Get current portfolio value
        port_value = cash
        for sym, shares in holdings.items():
            price = close_matrix.get(sym, {}).get(date, 0)
            if price > 0:
                port_value += shares * price

        equity_curve.append({
            "date": str(date)[:10],
            "equity": round(port_value, 2),
        })

        # Check if rebalance day
        if date in rebal_dates:
            rebalance_count += 1
            new_holdings, new_cash, turnover = _rebalance(
                date=date,
                price_data=price_data,
                fundamental_data=fundamental_data,
                engine=engine,
                sector_map=sector_map,
                current_holdings=holdings,
                current_cash=cash,
                portfolio_value=port_value,
                close_matrix=close_matrix,
                transaction_cost_pct=transaction_cost_pct,
            )
            total_turnover += turnover
            holdings = new_holdings
            cash = new_cash

        # Track monthly returns
        if i > 0 and bt_dates[i].month != bt_dates[i - 1].month:
            month_ret = (port_value / prev_equity - 1) * 100 if prev_equity > 0 else 0
            monthly_returns.append({
                "month": str(bt_dates[i - 1])[:7],
                "return_pct": round(month_ret, 2),
                "equity": round(port_value, 2),
            })
            prev_equity = port_value

    # ─── Step 4: Compute benchmark ────────────────────────────
    benchmark_equity = _compute_benchmark(
        price_data, bt_dates, initial_capital, benchmark_symbol,
    )

    # ─── Step 5: Compute metrics ──────────────────────────────
    result = _compute_metrics(
        factor_name=_engine_name(engine),
        equity_curve=equity_curve,
        monthly_returns=monthly_returns,
        benchmark_equity=benchmark_equity,
        initial_capital=initial_capital,
        total_turnover=total_turnover,
        rebalance_count=rebalance_count,
        top_n=engine.top_n,
    )

    return result


# ─── Internal Helpers ──────────────────────────────────────────


def _engine_name(engine: CompositeFactorEngine) -> str:
    names = [fc.factor.name for fc in engine.factors if fc.weight > 0]
    return " + ".join(names) if names else "Composite"


def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str,
) -> set[pd.Timestamp]:
    """Get the last trading day of each month/quarter."""
    rebal = set()
    for i in range(1, len(dates)):
        if freq == "monthly":
            if dates[i].month != dates[i - 1].month:
                rebal.add(dates[i - 1])
        elif freq == "quarterly":
            if dates[i].month != dates[i - 1].month and dates[i - 1].month in (3, 6, 9, 12):
                rebal.add(dates[i - 1])
    return rebal


def _build_close_matrix(
    price_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> dict[str, dict[pd.Timestamp, float]]:
    """Pre-compute close prices for fast lookup."""
    matrix = {}
    for sym, df in price_data.items():
        if "close" not in df.columns:
            continue
        closes = {}
        for date in dates:
            # Find closest date <= target
            mask = df.index <= date
            if mask.any():
                idx = df.index[mask][-1]
                closes[date] = float(df.loc[idx, "close"])
        if closes:
            matrix[sym] = closes
    return matrix


def _rebalance(
    date: pd.Timestamp,
    price_data: dict[str, pd.DataFrame],
    fundamental_data: dict[str, dict] | None,
    engine: CompositeFactorEngine,
    sector_map: dict[str, str],
    current_holdings: dict[str, int],
    current_cash: float,
    portfolio_value: float,
    close_matrix: dict[str, dict],
    transaction_cost_pct: float,
) -> tuple[dict[str, int], float, float]:
    """
    Execute a rebalance: rank → select → trade.
    Returns (new_holdings, new_cash, turnover_pct).
    """
    # Get target portfolio
    selected = engine.select_portfolio(
        price_data=price_data,
        fundamental_data=fundamental_data,
        sector_map=sector_map,
        as_of_date=date,
    )
    target_weights = engine.compute_weights(selected)

    if not target_weights:
        return current_holdings, current_cash, 0.0

    # Compute target shares
    target_holdings: dict[str, int] = {}
    for sym, weight in target_weights.items():
        alloc = portfolio_value * weight
        price = close_matrix.get(sym, {}).get(date, 0)
        if price > 0:
            target_holdings[sym] = int(alloc / price)

    # Compute trades (delta from current to target)
    all_symbols = set(list(current_holdings.keys()) + list(target_holdings.keys()))
    cash = current_cash
    turnover_value = 0.0
    new_holdings: dict[str, int] = {}

    for sym in all_symbols:
        current = current_holdings.get(sym, 0)
        target = target_holdings.get(sym, 0)
        delta = target - current
        price = close_matrix.get(sym, {}).get(date, 0)

        if price <= 0:
            if current > 0:
                new_holdings[sym] = current  # Can't trade, keep position
            continue

        trade_value = abs(delta) * price
        cost = trade_value * transaction_cost_pct

        if delta > 0:
            # Buy
            cash -= (delta * price + cost)
            turnover_value += trade_value
        elif delta < 0:
            # Sell
            cash += (abs(delta) * price - cost)
            turnover_value += trade_value

        if target > 0:
            new_holdings[sym] = target

    turnover_pct = turnover_value / portfolio_value if portfolio_value > 0 else 0
    return new_holdings, cash, turnover_pct


def _compute_benchmark(
    price_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    initial_capital: float,
    benchmark_symbol: str | None = None,
) -> list[dict]:
    """Compute benchmark equity curve (equal-weight buy-and-hold)."""
    equity = []

    if benchmark_symbol and benchmark_symbol in price_data:
        # Single stock/index benchmark
        df = price_data[benchmark_symbol]
        if "close" in df.columns and not df.empty:
            start_price = None
            for date in dates:
                mask = df.index <= date
                if mask.any():
                    price = float(df.loc[df.index[mask][-1], "close"])
                    if start_price is None:
                        start_price = price
                    val = initial_capital * (price / start_price)
                    equity.append({"date": str(date)[:10], "equity": round(val, 2)})
            return equity

    # Equal-weight universe buy-and-hold benchmark
    # Buy equal amounts of all stocks on day 1, hold forever
    symbols = list(price_data.keys())
    if not symbols:
        return []

    alloc = initial_capital / len(symbols)
    shares: dict[str, float] = {}

    # Buy at start
    for sym in symbols:
        df = price_data[sym]
        if "close" not in df.columns:
            continue
        mask = df.index <= dates[0]
        if mask.any():
            price = float(df.loc[df.index[mask][-1], "close"])
            if price > 0:
                shares[sym] = alloc / price

    if not shares:
        return []

    for date in dates:
        val = 0.0
        for sym, sh in shares.items():
            df = price_data[sym]
            mask = df.index <= date
            if mask.any():
                price = float(df.loc[df.index[mask][-1], "close"])
                val += sh * price
        equity.append({"date": str(date)[:10], "equity": round(val, 2)})

    return equity


def _compute_metrics(
    factor_name: str,
    equity_curve: list[dict],
    monthly_returns: list[dict],
    benchmark_equity: list[dict],
    initial_capital: float,
    total_turnover: float,
    rebalance_count: int,
    top_n: int,
) -> FactorBacktestResult:
    """Compute all backtest metrics from equity curve."""

    if not equity_curve:
        return FactorBacktestResult(factor_name=factor_name, start_date="", end_date="")

    start_date = equity_curve[0]["date"]
    end_date = equity_curve[-1]["date"]
    final_equity = equity_curve[-1]["equity"]

    # Total return
    total_return = (final_equity / initial_capital - 1) * 100

    # CAGR
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    years = days / 365.25
    cagr = ((final_equity / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0

    # Daily returns for Sharpe
    equities = [e["equity"] for e in equity_curve]
    daily_returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            daily_returns.append(equities[i] / equities[i - 1] - 1)

    sharpe = 0.0
    if len(daily_returns) > 30:
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        if std_r > 0:
            sharpe = round(mean_r / std_r * np.sqrt(252), 2)

    # Max drawdown
    peak = initial_capital
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    # Benchmark metrics
    benchmark_cagr = 0.0
    if benchmark_equity:
        bm_final = benchmark_equity[-1]["equity"]
        benchmark_cagr = ((bm_final / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0

    alpha = cagr - benchmark_cagr

    # Information ratio (alpha / tracking error)
    ir = 0.0
    if benchmark_equity and len(benchmark_equity) == len(equity_curve):
        excess_returns = []
        for i in range(1, len(equity_curve)):
            port_ret = equities[i] / equities[i - 1] - 1
            bm_ret = benchmark_equity[i]["equity"] / benchmark_equity[i - 1]["equity"] - 1
            excess_returns.append(port_ret - bm_ret)
        if len(excess_returns) > 30:
            te = np.std(excess_returns) * np.sqrt(252)
            if te > 0:
                ir = round(np.mean(excess_returns) * 252 / (te * np.sqrt(252)), 2)

    # Hit rate (% of months beating benchmark)
    hit_rate = 0.0
    if monthly_returns and benchmark_equity:
        bm_monthly = _monthly_benchmark_returns(benchmark_equity)
        hits = 0
        total = 0
        for mr in monthly_returns:
            bm_r = bm_monthly.get(mr["month"], 0)
            total += 1
            if mr["return_pct"] >= bm_r:
                hits += 1
        hit_rate = (hits / total * 100) if total > 0 else 0

    # Yearly returns
    yearly = _compute_yearly_returns(equity_curve, initial_capital)

    # Average turnover
    avg_turnover = (total_turnover / max(rebalance_count, 1)) * 100

    return FactorBacktestResult(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        sharpe_ratio=sharpe,
        max_drawdown_pct=round(max_dd, 2),
        avg_monthly_turnover=round(avg_turnover, 1),
        avg_holdings=top_n,
        total_rebalances=rebalance_count,
        benchmark_cagr_pct=round(benchmark_cagr, 2),
        alpha_pct=round(alpha, 2),
        information_ratio=ir,
        hit_rate_monthly=round(hit_rate, 1),
        equity_curve=equity_curve,
        monthly_returns=monthly_returns,
        yearly_returns=yearly,
    )


def _monthly_benchmark_returns(benchmark: list[dict]) -> dict[str, float]:
    """Extract monthly returns from benchmark equity curve."""
    monthly = {}
    prev = benchmark[0]["equity"]
    prev_month = benchmark[0]["date"][:7]
    for entry in benchmark[1:]:
        month = entry["date"][:7]
        if month != prev_month:
            ret = (entry["equity"] / prev - 1) * 100
            monthly[prev_month] = round(ret, 2)
            prev = entry["equity"]
            prev_month = month
    return monthly


def _compute_yearly_returns(
    equity_curve: list[dict],
    initial_capital: float,
) -> list[dict]:
    """Compute year-by-year returns."""
    yearly = {}
    for entry in equity_curve:
        year = entry["date"][:4]
        if year not in yearly:
            yearly[year] = {"first": entry["equity"], "last": entry["equity"]}
        yearly[year]["last"] = entry["equity"]

    results = []
    for year in sorted(yearly.keys()):
        first = yearly[year]["first"]
        last = yearly[year]["last"]
        ret = (last / first - 1) * 100 if first > 0 else 0
        results.append({"year": year, "return_pct": round(ret, 2)})
    return results


# ─── CLI Entry Point ──────────────────────────────────────────

def main():
    """Run factor backtest from command line."""
    import argparse
    from factors.composite import PRESETS
    from factors.universe import get_universe, fetch_universe_prices, load_fundamental_data

    parser = argparse.ArgumentParser(description="Factor Engine Backtest")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="momentum_quality")
    parser.add_argument("--universe", choices=["nifty50", "nifty100", "nifty200"], default="nifty200")
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--freq", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--weighting", choices=["equal", "score_weighted"], default="equal")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  FACTOR ENGINE BACKTEST")
    print(f"  Preset: {args.preset} | Universe: {args.universe}")
    print(f"  Top N: {args.top_n} | Freq: {args.freq} | Capital: ₹{args.capital:,.0f}")
    print(f"{'='*70}\n")

    # Fetch data
    symbols = get_universe(args.universe)
    print(f"  Universe: {len(symbols)} stocks")

    price_data = fetch_universe_prices(symbols, years=args.years)
    fund_data = load_fundamental_data()

    # Build engine
    factors = PRESETS[args.preset]()
    engine = CompositeFactorEngine(
        factors=factors,
        top_n=args.top_n,
        weighting=args.weighting,
    )

    # Run backtest
    result = run_factor_backtest(
        price_data=price_data,
        engine=engine,
        fundamental_data=fund_data if fund_data else None,
        rebalance_freq=args.freq,
        initial_capital=args.capital,
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS: {result.factor_name}")
    print(f"{'='*70}")
    print(f"  Period:       {result.start_date} → {result.end_date}")
    print(f"  Total Return: {result.total_return_pct:+.1f}%")
    print(f"  CAGR:         {result.cagr_pct:+.1f}%")
    print(f"  Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"  Avg Turnover: {result.avg_monthly_turnover:.1f}%/rebal")
    print(f"  Rebalances:   {result.total_rebalances}")
    print(f"  ─────────────────────────────────")
    print(f"  Benchmark:    {result.benchmark_cagr_pct:+.1f}% CAGR")
    print(f"  ALPHA:        {result.alpha_pct:+.1f}%")
    print(f"  Info Ratio:   {result.information_ratio:.2f}")
    print(f"  Hit Rate:     {result.hit_rate_monthly:.0f}% months beating benchmark")

    if result.yearly_returns:
        print(f"\n  Year-by-year:")
        for yr in result.yearly_returns:
            print(f"    {yr['year']}: {yr['return_pct']:+.1f}%")

    # Save results
    import json
    out_path = "data/factor_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
