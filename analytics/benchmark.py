"""
analytics/benchmark.py

Benchmark layer for comparing strategy / portfolio results against
simple passive baselines.

Three benchmarks:
1. NIFTY 50 Proxy — uses real index data if available, else equal-weight proxy
2. Equal-Weight Basket — static equal capital allocation across symbols
3. Buy & Hold Basket — same as equal-weight (static allocation, held through period)

All benchmarks share one reusable core implementation since Equal-Weight and
Buy & Hold are identical for the static case (allocate once at start, hold).

Pure functions — no side effects, no network calls.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analytics.drawdown import max_drawdown


# ──────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Result of a single benchmark computation."""

    label: str
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sharpe_valid: bool = False
    equity_curve: list[tuple[date, float]] = field(default_factory=list)


# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────

NIFTY_PROXY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
]

# ──────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────


def _compute_cagr(initial: float, final: float, years: float) -> float:
    """Compound annual growth rate (%)."""
    if years <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return round(((final / initial) ** (1 / years) - 1) * 100, 2)


def _compute_sharpe(
    equities: list[float], min_days: int = 30
) -> tuple[float, bool]:
    """Sharpe ratio from daily equity-curve returns."""
    if len(equities) < min_days + 1:
        return 0.0, False
    returns = np.diff(equities) / np.array(equities[:-1])
    if len(returns) == 0 or np.std(returns) == 0 or np.isnan(np.std(returns)):
        return 0.0, False
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    return round(sharpe, 2), True


def _build_date_union(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Build sorted union of trading dates across provided symbols."""
    all_dates: set[pd.Timestamp] = set()
    for sym in symbols:
        if sym in data:
            all_dates.update(data[sym].index)

    dates = sorted(all_dates)
    if start_date:
        dates = [d for d in dates if d >= pd.to_datetime(start_date)]
    if end_date:
        dates = [d for d in dates if d <= pd.to_datetime(end_date)]
    return dates


# ──────────────────────────────────────────
# Core benchmark implementation
# ──────────────────────────────────────────


def _run_static_basket(
    label: str,
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    initial_capital: float = 100_000,
) -> BenchmarkResult:
    """
    Core benchmark implementation.

    Allocates capital equally across valid symbols at the start of the period.
    Buys shares at the first available close price for each symbol.
    Holds throughout (no rebalancing).
    Marks to market daily using close prices.

    Used by both `run_equal_weight` and `run_buy_and_hold`.
    """
    # Filter to symbols that have data
    valid_symbols = [s for s in symbols if s in data and not data[s].empty]

    if not valid_symbols:
        return BenchmarkResult(label=label, equity_curve=[])

    dates = _build_date_union(data, valid_symbols, start_date, end_date)
    if len(dates) < 2:
        return BenchmarkResult(label=label, equity_curve=[])

    # Allocate equally at first trading date
    first_date = dates[0]
    capital_per_sym = initial_capital / len(valid_symbols)

    holdings: dict[str, dict] = {}  # sym -> {'shares': int, 'entry_price': float}
    entry_prices: dict[str, float] = {}

    for sym in valid_symbols:
        df = data[sym]
        if first_date not in df.index:
            continue
        price = float(df.loc[first_date, "close"])
        if price <= 0:
            continue
        shares = int(capital_per_sym / price)
        if shares > 0:
            holdings[sym] = {"shares": shares, "entry_price": price}
            entry_prices[sym] = price

    if not holdings:
        return BenchmarkResult(label=label, equity_curve=[])

    # Track equity daily
    equity_curve: list[tuple[date, float]] = []

    for d in dates:
        total = 0.0
        for sym, h in holdings.items():
            if d in data[sym].index:
                price = float(data[sym].loc[d, "close"])
                total += h["shares"] * price
            else:
                # Use last known price (carry forward)
                total += h["shares"] * h["entry_price"]
        equity_curve.append((d.date(), round(total, 2)))

    equities = [e for _, e in equity_curve]
    final_equity = equities[-1] if equities else initial_capital
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)

    cagr = _compute_cagr(initial_capital, final_equity, years)
    max_dd = max_drawdown(equities) if equities else 0.0
    sharpe, sharpe_valid = _compute_sharpe(equities) if equities else (0.0, False)
    total_return_pct = round((final_equity / initial_capital - 1) * 100, 2)

    return BenchmarkResult(
        label=label,
        final_equity=round(final_equity, 2),
        total_return_pct=total_return_pct,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sharpe_valid=sharpe_valid,
        equity_curve=equity_curve,
    )


# ──────────────────────────────────────────
# Public API — pure functions
# ──────────────────────────────────────────


def run_buy_and_hold(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    initial_capital: float = 100_000,
) -> BenchmarkResult:
    """
    Buy & Hold benchmark.

    Allocates capital equally across symbols at start date.
    Holds all positions through the entire period.
    No rebalancing, no exits.

    Notes
    -----
    Identical to equal-weight static allocation (no rebalancing).
    If you need a meaningfully different third benchmark, consider:
    - Buy & Hold with a trend filter (e.g. above 200SMA)
    - A single large-cap proxy (e.g. 100% RELIANCE)
    """
    return _run_static_basket(
        label="Buy & Hold",
        data=data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )


def run_equal_weight(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    initial_capital: float = 100_000,
) -> BenchmarkResult:
    """
    Equal-Weight benchmark.

    Same as Buy & Hold for static allocation (allocate once, hold through).
    For true equal-weight with rebalancing, extend this function.

    Returns the same result as `run_buy_and_hold` for the static case.
    """
    return _run_static_basket(
        label="Equal Weight",
        data=data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )


def run_nifty_proxy(
    data: dict[str, pd.DataFrame],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    initial_capital: float = 100_000,
    proxy_symbols: list[str] | None = None,
) -> BenchmarkResult:
    """
    NIFTY 50 Proxy benchmark.

    Uses real index data if available (keyed as '^NSEI' or 'NIFTY50' in data),
    else falls back to an equal-weight proxy basket.

    Default proxy symbols: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
    """
    if proxy_symbols is None:
        proxy_symbols = list(NIFTY_PROXY_SYMBOLS)

    # Try real index data first
    for index_key in ("^NSEI", "NIFTY50", "NIFTY 50"):
        if index_key in data and not data[index_key].empty:
            df = data[index_key]
            dates = _build_date_union(data, [index_key], start_date, end_date)
            if len(dates) >= 2:
                return _run_static_basket(
                    label="NIFTY50 Proxy",
                    data={index_key: df},
                    symbols=[index_key],
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                )

    # Fallback: equal-weight proxy basket
    return _run_static_basket(
        label="NIFTY50 Proxy",
        data=data,
        symbols=proxy_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )


def compare_results(
    strategy_cagr: float,
    strategy_maxdd: float,
    benchmarks: list[BenchmarkResult],
) -> tuple[list[dict[str, Any]], str]:
    """
    Compare strategy results against a list of benchmarks.

    Parameters
    ----------
    strategy_cagr : float
        Strategy CAGR percentage.
    strategy_maxdd : float
        Strategy max drawdown percentage.
    benchmarks : list[BenchmarkResult]
        Benchmark results to compare against.

    Returns
    -------
    comparisons : list[dict]
        Each dict: label, benchmark_cagr, benchmark_maxdd, cagr_diff, dd_diff.
    verdict : str
        Summary verdict on strategy vs benchmarks.
    """
    comparisons = []
    for b in benchmarks:
        comparisons.append({
            "label": b.label,
            "benchmark_cagr": b.cagr,
            "benchmark_maxdd": b.max_drawdown,
            "cagr_diff": round(strategy_cagr - b.cagr, 1),
            "dd_diff": round(strategy_maxdd - b.max_drawdown, 1),
        })

    # Simple verdict
    outperformed = sum(1 for c in comparisons if c["cagr_diff"] > 0)
    total = len(comparisons)
    if outperformed >= total / 2:
        verdict = f"Strategy outperforms {outperformed}/{total} benchmarks on CAGR"
    else:
        verdict = f"Strategy underperforms {total - outperformed}/{total} benchmarks on CAGR"

    return comparisons, verdict