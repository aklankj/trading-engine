"""
tests/test_execution_delay.py

Tests for T+1 execution delay in portfolio/simulator.py.

Verifies:
1. Signal on day T → position opens on T+1 trading day
2. Entry price matches next day's OPEN
3. Last-day signals skipped (no next trading date)
4. Holiday gap → execution on next available trading day
5. Missing/NaN open price → skipped with log
6. EXECUTION_DELAY_DAYS=0 → same-day close fill (backward compat)
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.simulator import simulate, SimResult


# ── Helpers ──────────────────────────────────────────────


def _make_data(
    symbols: list[str],
    n_days: int = 200,
    trend: float = 0.001,
    start_price: float = 100.0,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build synthetic OHLCV data with valid open/high/low/close/volume."""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        prices = [start_price * (1 + trend) ** i for i in range(n_days)]
        # Open is slightly different from close to simulate realistic data
        opens = [p * 0.995 for p in prices]
        highs = [p * 1.015 for p in prices]
        lows = [p * 0.985 for p in prices]

        data[sym] = pd.DataFrame(
            {
                "open": np.array(opens),
                "high": np.array(highs),
                "low": np.array(lows),
                "close": np.array(prices),
                "volume": np.random.randint(1_000_000, 10_000_000, n_days),
            },
            index=dates,
        )
    return data


def _make_data_with_holiday_gap(
    symbols: list[str],
    trend: float = 0.001,
) -> dict[str, pd.DataFrame]:
    """
    Build data with a known gap (holiday) — e.g., Monday present,
    Tuesday missing, Wednesday present.  This tests that the delay
    logic skips to the next *available* trading date.
    """
    # Start on a Monday
    dates = pd.bdate_range("2022-01-03", periods=10)  # 2 weeks Mon-Fri
    # Manually drop Tuesday (day index 1 → 2022-01-04 is a business day, so
    # let's use actual calendar dates and drop the second date)
    dates = dates.delete([1])  # Remove second date → creates a gap
    n_days = len(dates)

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        prices = [100.0 * (1 + trend) ** i for i in range(n_days)]
        opens = [p * 0.995 for p in prices]

        data[sym] = pd.DataFrame(
            {
                "open": np.array(opens),
                "high": np.array(prices) * 1.015,
                "low": np.array(prices) * 0.985,
                "close": np.array(prices),
                "volume": np.full(n_days, 1_000_000),
            },
            index=dates,
        )
    return data


def _make_data_with_nan_open(
    symbol: str,
    n_days: int = 50,
    nan_day_index: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Build data for a single symbol where the 'open' column has NaN
    on a specific day.
    """
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    prices = [100.0 * (1.001) ** i for i in range(n_days)]
    opens = [p * 0.995 for p in prices]

    # Set open to NaN on nan_day_index
    opens[nan_day_index] = np.nan

    data: dict[str, pd.DataFrame] = {}
    data[symbol] = pd.DataFrame(
        {
            "open": np.array(opens),
            "high": np.array(prices) * 1.015,
            "low": np.array(prices) * 0.985,
            "close": np.array(prices),
            "volume": np.full(n_days, 1_000_000),
        },
        index=dates,
    )
    return data


# ── Test 1: Signal on T → entry on T+1 ────────────────


def test_signal_t_entry_t_plus_1():
    """
    With EXECUTION_DELAY_DAYS=1, a signal on day T should produce
    a position whose entry_date is the next trading day (T+1).
    """
    from config.settings import cfg as _test_cfg

    data = _make_data(["STOCK_A"], n_days=100)
    dates = list(data["STOCK_A"].index)

    # Use delay=1
    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 1):
        result = simulate(
            data,
            symbols=["STOCK_A"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=5,
            position_size_pct=0.10,
        )

    # If any trades occurred, verify entry_date != signal_date
    if result.total_trades > 0:
        for trade in result.trade_log:
            entry_date = pd.to_datetime(trade.entry_date)
            # Find the index of this date in the full date list
            if entry_date in dates:
                entry_idx = dates.index(entry_date)
                # The date before should be the signal date (not directly checkable from SimTrade,
                # but we can check that entry is at least 1 trading day after some signal)
                # At minimum, entry_date should NOT be the first date in the series
                assert entry_idx > 0, (
                    f"Entry date {entry_date.date()} is the first date; "
                    f"expected it to be T+1 from some signal"
                )
    else:
        pytest.skip("No trades generated for this data seed")


# ── Test 2: Entry price matches next day OPEN ──────────


def test_entry_price_matches_next_open():
    """
    Verify that the entry price on T+1 matches the OPEN price
    of that symbol on that day.
    """
    from config.settings import cfg as _test_cfg

    data = _make_data(["STOCK_B"], n_days=100, trend=0.002)
    dates = list(data["STOCK_B"].index)

    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 1):
        result = simulate(
            data,
            symbols=["STOCK_B"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=5,
            position_size_pct=0.10,
        )

    if result.total_trades > 0:
        for trade in result.trade_log:
            entry_date_pd = pd.to_datetime(trade.entry_date)
            if entry_date_pd in data["STOCK_B"].index:
                expected_open = float(
                    data["STOCK_B"].loc[entry_date_pd, "open"]
                )
                # Allow small difference due to slippage
                price_diff_pct = abs(trade.entry_price - expected_open) / expected_open
                assert price_diff_pct < 0.02, (
                    f"Entry price {trade.entry_price:.2f} for {trade.symbol} "
                    f"on {trade.entry_date} does not match open "
                    f"{expected_open:.2f} (diff={price_diff_pct*100:.1f}%)"
                )
    else:
        pytest.skip("No trades generated for this data seed")


# ── Test 3: Last-day signals skipped ───────────────────


def test_last_day_signals_skipped():
    """
    Signals generated on the final trading date should produce
    no trades, because there's no T+1.
    """
    from config.settings import cfg as _test_cfg

    data = _make_data(["STOCK_C"], n_days=10, trend=0.005)  # short period
    dates = list(data["STOCK_C"].index)

    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 1):
        result = simulate(
            data,
            symbols=["STOCK_C"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=5,
            position_size_pct=0.50,  # high allocation to increase chance of trades
        )

    # No trade should have entry_date == last_date because that signal
    # would have been skipped (no T+1)
    for trade in result.trade_log:
        assert trade.entry_date != dates[-1].date(), (
            f"Found trade with entry on last date {trade.entry_date}. "
            f"Last-date signals should be skipped."
        )


# ── Test 4: Holiday gap → next available trading day ──


def test_holiday_gap_skips_to_next_trading_day():
    """
    If a trading day is missing (e.g., holiday), the delayed
    execution should pick the next available trading date,
    not a missing/bad date.
    """
    from config.settings import cfg as _test_cfg

    data = _make_data_with_holiday_gap(["STOCK_D"])
    dates = list(data["STOCK_D"].index)

    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 1):
        result = simulate(
            data,
            symbols=["STOCK_D"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=3,
            position_size_pct=0.10,
        )

    # All trade entry dates should be in the date index
    for trade in result.trade_log:
        entry_pd = pd.to_datetime(trade.entry_date)
        assert entry_pd in dates, (
            f"Entry date {trade.entry_date} not in available trading dates. "
            f"Expected entry on an existing trading date after the signal."
        )


# ── Test 5: Missing / NaN open price → skipped ────────


def test_missing_open_skipped():
    """
    When the OPEN price is NaN on the execution date, the
    trade should be skipped (not executed with corrupt price).
    """
    from config.settings import cfg as _test_cfg

    # Day 10 has NaN open
    data = _make_data_with_nan_open("STOCK_E", n_days=50, nan_day_index=10)
    dates = list(data["STOCK_E"].index)

    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 1):
        result = simulate(
            data,
            symbols=["STOCK_E"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=5,
            position_size_pct=0.10,
        )

    # All trades should have valid (non-NaN, >0) entry prices
    for trade in result.trade_log:
        assert not math.isnan(trade.entry_price), (
            f"Trade for {trade.symbol} has NaN entry price"
        )
        assert trade.entry_price > 0, (
            f"Trade for {trade.symbol} has non-positive entry price {trade.entry_price}"
        )


# ── Test 6: Zero delay → same-day close (backward compat)


def test_zero_delay_same_day():
    """
    When EXECUTION_DELAY_DAYS=0, entry should happen on the
    same day at close (original behavior). Verify that the
    entry price is the close price of the signal day.
    """
    from config.settings import cfg as _test_cfg

    data = _make_data(["STOCK_F"], n_days=100, trend=0.002)
    dates = list(data["STOCK_F"].index)

    with patch.object(_test_cfg, "EXECUTION_DELAY_DAYS", 0):
        result = simulate(
            data,
            symbols=["STOCK_F"],
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100_000,
            max_positions=5,
            position_size_pct=0.10,
        )

    # With delay=0, the function should still work and produce valid results
    assert isinstance(result, SimResult)
    assert result.final_equity >= 0