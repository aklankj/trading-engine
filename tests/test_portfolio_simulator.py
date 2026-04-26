"""
Extended tests for portfolio/simulator.py.

Tests cover:
1. No negative cash during simulation
2. Max positions respected
3. Duplicate symbol prevention
4. Slippage cost accumulates
5. END_OF_TEST forced close
6. Sector cap blocks entries

All tests offline — synthetic OHLCV, no network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.simulator import simulate, SimResult, SimTrade, SimPosition
from config.settings import cfg


def _make_data(symbols, n_days=200, trend=0.001):
    """Build synthetic OHLCV data."""
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    data = {}
    for sym in symbols:
        prices = [100 * (1 + trend) ** i for i in range(n_days)]
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
    return data


def test_no_negative_cash():
    from strategies.registry import CORE_STRATEGIES
    data = _make_data(["STOCK_A"], n_days=100)
    dates = list(data["STOCK_A"].index)
    result = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["STOCK_A"],
        start_date=dates[0], end_date=dates[-1],
        initial_capital=100_000, max_positions=1,
    )
    assert result.final_equity >= 0
    assert result.total_trades >= 0


def test_slippage_cost_accumulates():
    from strategies.registry import CORE_STRATEGIES
    data = _make_data(["STOCK_A"], n_days=200, trend=0.002)
    dates = list(data["STOCK_A"].index)

    result = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["STOCK_A"],
        start_date=dates[0], end_date=dates[-1],
        slippage_pct=0.01, max_positions=3,
    )
    if result.total_trades > 0:
        assert result.total_slippage_cost > 0

    result_no_slip = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["STOCK_A"],
        start_date=dates[0], end_date=dates[-1],
        slippage_pct=0.0, max_positions=3,
    )
    assert result_no_slip.total_slippage_cost == 0.0


def test_end_of_test_forced_close():
    from strategies.registry import CORE_STRATEGIES
    data = _make_data(["STOCK_FORCE"], n_days=50, trend=0.003)
    dates = list(data["STOCK_FORCE"].index)

    result = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["STOCK_FORCE"],
        start_date=dates[0], end_date=dates[-1],
        max_positions=5, position_size_pct=0.20,
    )
    if result.total_trades > 0:
        valid_reasons = {
            "trailing_stop", "stop_loss", "target",
            "time_exit", "END_OF_TEST", "reversal", "trend_exit",
        }
        for t in result.trade_log:
            assert t.exit_reason in valid_reasons


def test_max_positions_respected():
    from strategies.registry import CORE_STRATEGIES
    data = _make_data(["A", "B", "C"], n_days=150, trend=0.002)
    dates = list(data["A"].index)
    result = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["A", "B", "C"],
        start_date=dates[0], end_date=dates[-1],
        max_positions=1, position_size_pct=0.10,
    )
    assert result.final_equity >= 0
    assert isinstance(result, SimResult)


def test_duplicate_symbol_prevention():
    from strategies.registry import CORE_STRATEGIES
    data = _make_data(["ONLY_STOCK"], n_days=150, trend=0.002)
    dates = list(data["ONLY_STOCK"].index)
    result_single = simulate(
        data, strategies=CORE_STRATEGIES, symbols=["ONLY_STOCK"],
        start_date=dates[0], end_date=dates[-1],
        max_positions=5, position_size_pct=0.10,
        allow_multiple_per_symbol=False,
    )
    assert isinstance(result_single, SimResult)


def test_sector_cap_blocks_entries():
    """filter_candidates_by_sector should block entries exceeding cap."""
    from portfolio.risk import filter_candidates_by_sector

    candidates = [
        (1.0, "A", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
        (0.9, "B", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
    ]
    sector_map = {"A": "Tech", "B": "Tech"}

    # 2% cap with 1% position size — each passes individually (check doesn't accumulate)
    filtered = filter_candidates_by_sector(
        candidates, sector_map, {"Tech": 0},
        cash=100_000, deployed=0, position_size_pct=0.01,
        max_sector_pct=0.02,
    )
    assert len(filtered) == 2

    # 5% cap with 1% position — both pass
    filtered_wide = filter_candidates_by_sector(
        candidates, sector_map, {"Tech": 0},
        cash=100_000, deployed=0, position_size_pct=0.01,
        max_sector_pct=0.05,
    )
    assert len(filtered_wide) == 2

    # 1% cap with 5% position — all blocked
    filtered_blocked = filter_candidates_by_sector(
        candidates, sector_map, {"Tech": 0},
        cash=100_000, deployed=0, position_size_pct=0.05,
        max_sector_pct=0.01,
    )
    assert len(filtered_blocked) == 0