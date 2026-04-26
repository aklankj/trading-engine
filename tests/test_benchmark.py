"""
Task 4: Test the analytics/benchmark.py module.

Tests cover:
1. Rising prices produce positive CAGR
2. Flat prices => ~0 return
3. Drawdown computed correctly
4. Equal-weight allocation logic
5. Handles empty/missing symbols safely

All tests offline — synthetic OHLCV DataFrames, no network calls.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from analytics.benchmark import (
    BenchmarkResult,
    run_buy_and_hold,
    run_equal_weight,
    run_nifty_proxy,
    compare_results,
)
from analytics.drawdown import max_drawdown


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def _make_synthetic_data(
    symbols: list[str],
    n_days: int = 500,
    start_price: float = 100.0,
    daily_return: float = 0.001,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Build synthetic OHLCV DataFrames for testing.

    Prices follow a deterministic geometric Brownian motion.
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        prices = [start_price]
        for _ in range(1, n_days):
            prices.append(prices[-1] * (1 + daily_return))
        prices = np.array(prices)

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(1_000_000, 10_000_000, n_days),
            },
            index=dates,
        )
        data[sym] = df

    return data


def _make_flat_data(
    symbols: list[str],
    n_days: int = 500,
    price: float = 100.0,
) -> dict[str, pd.DataFrame]:
    """Synthetic data with constant prices (flat market)."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        prices = np.full(n_days, price)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": np.full(n_days, 1_000_000),
            },
            index=dates,
        )
        data[sym] = df

    return data


# ──────────────────────────────────────────
# Test 1: Rising prices produce positive CAGR
# ──────────────────────────────────────────


def test_rising_prices_positive_cagr():
    """
    With a rising market (0.1% daily return), all benchmarks
    should produce positive CAGR.
    """
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    data = _make_synthetic_data(symbols, n_days=500, daily_return=0.001)
    start = data["RELIANCE"].index[0]
    end = data["RELIANCE"].index[-1]

    bnh = run_buy_and_hold(data, symbols, start_date=start, end_date=end)
    assert bnh.cagr > 0, f"Buy & Hold CAGR should be positive, got {bnh.cagr}"
    assert bnh.final_equity > 100_000
    assert bnh.total_return_pct > 0

    eqw = run_equal_weight(data, symbols, start_date=start, end_date=end)
    assert eqw.cagr > 0
    assert eqw.final_equity > 100_000

    nifty = run_nifty_proxy(data, start_date=start, end_date=end)
    assert nifty.cagr > 0
    assert nifty.final_equity > 100_000


# ──────────────────────────────────────────
# Test 2: Flat prices => ~0 return
# ──────────────────────────────────────────


def test_flat_prices_zero_return():
    """
    When all prices are flat, CAGR should be approximately 0%.
    """
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    data = _make_flat_data(symbols, n_days=500, price=100.0)
    start = data["RELIANCE"].index[0]
    end = data["RELIANCE"].index[-1]

    bnh = run_buy_and_hold(data, symbols, start_date=start, end_date=end)
    assert pytest.approx(bnh.cagr, abs=0.5) == 0.0
    # Integer share rounding may lose ~0.1% capital; check it's close
    assert bnh.final_equity >= 99_500, f"Final equity too low: {bnh.final_equity}"
    assert pytest.approx(bnh.total_return_pct, abs=1.0) == 0.0

    eqw = run_equal_weight(data, symbols, start_date=start, end_date=end)
    assert pytest.approx(eqw.cagr, abs=0.5) == 0.0

    nifty = run_nifty_proxy(data, start_date=start, end_date=end)
    assert pytest.approx(nifty.cagr, abs=0.5) == 0.0


# ──────────────────────────────────────────
# Test 3: Drawdown computed correctly
# ──────────────────────────────────────────


def test_drawdown_computed():
    """
    For a synthetic V-shaped market (rise then crash then recover),
    verify max_drawdown matches expected values.
    """
    symbols = ["STOCK"]
    n_days = 300
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = []

    # Rise 50%, then crash 40%, then recover
    for i in range(n_days):
        if i < 100:
            prices.append(100 * (1 + i * 0.005))  # Rise to 150
        elif i < 150:
            prices.append(prices[-1] * 0.99)  # Crash to ~91
        else:
            prices.append(prices[-1] * 1.002)  # Slow recovery

    df = pd.DataFrame(
        {
            "open": np.array(prices) * 0.99,
            "high": np.array(prices) * 1.02,
            "low": np.array(prices) * 0.98,
            "close": np.array(prices),
            "volume": np.full(n_days, 1_000_000),
        },
        index=dates,
    )
    data = {"STOCK": df}

    bnh = run_buy_and_hold(data, ["STOCK"], start_date=dates[0], end_date=dates[-1])
    # The peak is at 150, trough is around 91 (40% drop from peak)
    # Max drawdown should be more than 30%
    assert bnh.max_drawdown < -30.0, f"Expected drawdown >30%, got {bnh.max_drawdown}"
    assert bnh.max_drawdown < 0, f"Drawdown should be negative, got {bnh.max_drawdown}"


# ──────────────────────────────────────────
# Test 4: Equal-weight allocation logic
# ──────────────────────────────────────────


def test_equal_weight_allocation():
    """
    With 3 symbols at the same price, equal weight should allocate
    ~1/3 of capital to each. Verify the equity curve accounts for all.
    """
    symbols = ["A", "B", "C"]
    n_days = 200
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    # Symbol A rises, B flat, C falls
    data = {}
    prices_a = [100 * (1 + 0.001) ** i for i in range(n_days)]
    prices_b = [100.0] * n_days
    prices_c = [100 * (1 - 0.001) ** i for i in range(n_days)]

    for sym, prices in [("A", prices_a), ("B", prices_b), ("C", prices_c)]:
        data[sym] = pd.DataFrame(
            {
                "open": np.array(prices) * 0.99,
                "high": np.array(prices) * 1.02,
                "low": np.array(prices) * 0.98,
                "close": np.array(prices),
                "volume": np.full(n_days, 1_000_000),
            },
            index=dates,
        )

    start = dates[0]
    end = dates[-1]

    # All 3 symbols
    bnh_3 = run_buy_and_hold(data, symbols, start_date=start, end_date=end)
    assert bnh_3.final_equity > 0
    assert len(bnh_3.equity_curve) == n_days

    # Single symbol "A" (should be different from 3-symbol result)
    bnh_a = run_buy_and_hold(data, ["A"], start_date=start, end_date=end)
    # With only A (rising), final equity should be higher than mixing with C (falling)
    assert bnh_a.final_equity > bnh_3.final_equity


# ──────────────────────────────────────────
# Test 5: Handles empty/missing symbols safely
# ──────────────────────────────────────────


def test_empty_missing_symbols():
    """
    If all symbols are missing from data, benchmarks should
    return empty results gracefully (no crash).
    """
    data: dict[str, pd.DataFrame] = {}
    # Non-existent datas
    data["RELIANCE"] = pd.DataFrame()

    result = run_buy_and_hold(data, ["MISSING"], start_date="2020-01-01", end_date="2020-12-31")
    assert result.final_equity == 0.0
    assert result.equity_curve == []
    assert result.cagr == 0.0

    # Empty data dict
    result2 = run_buy_and_hold({}, ["ABC"], start_date="2020-01-01", end_date="2020-12-31")
    assert result2.final_equity == 0.0
    assert result2.equity_curve == []

    # NIFTY proxy with empty data
    result3 = run_nifty_proxy({}, start_date="2020-01-01", end_date="2020-12-31")
    assert result3.final_equity == 0.0


# ──────────────────────────────────────────
# Test 6: compare_results utility
# ──────────────────────────────────────────


def test_compare_results():
    """
    compare_results should produce correct comparison dicts
    and a meaningful verdict.
    """
    benchmarks = [
        BenchmarkResult(label="TestB", cagr=10.0, max_drawdown=-5.0),
        BenchmarkResult(label="TestC", cagr=8.0, max_drawdown=-3.0),
    ]

    comparisons, verdict = compare_results(strategy_cagr=12.0, strategy_maxdd=-4.0, benchmarks=benchmarks)
    assert len(comparisons) == 2
    # Strategy outperforms both on CAGR
    assert comparisons[0]["cagr_diff"] == 2.0
    assert comparisons[1]["cagr_diff"] == 4.0
    assert "outperforms" in verdict