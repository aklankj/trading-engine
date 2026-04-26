"""Tests for backtest/reporting.py — printing utilities."""

from __future__ import annotations

from unittest.mock import patch
from io import StringIO

import pytest

from backtest.reporting import (
    print_backtest_header,
    print_backtest_table,
    print_backtest_footer,
    print_compact_summary,
    print_benchmarks,
    print_walkforward_header,
    print_walkforward_table,
    print_walkforward_summary,
)


def _mock_summary():
    """Return a minimal summary dict for testing."""
    from strategies.base import BacktestResult
    return {
        "TestStrat": BacktestResult(
            strategy="TestStrat",
            total_trades=10,
            winners=6,
            losers=4,
            win_rate=60.0,
            cagr=12.5,
            expectancy=2.1,
            sharpe=1.5,
            sharpe_valid=True,
            profit_factor=2.0,
            max_drawdown=-15.0,
            avg_win=5.0,
            avg_loss=-3.0,
            avg_hold_days=30,
        ),
        "EmptyStrat": BacktestResult(strategy="EmptyStrat"),
    }


def _mock_benchmark_result(label, cagr=8.0, max_dd=-10.0):
    """Create a mock benchmark result."""
    from dataclasses import dataclass, field
    @dataclass
    class MockBenchmark:
        label: str
        cagr: float
        max_drawdown: float
        equity_curve: list = field(default_factory=lambda: [(1, 1)])
    return MockBenchmark(label=label, cagr=cagr, max_drawdown=max_dd)


def test_print_backtest_table_does_not_crash():
    """print_backtest_table should handle empty results gracefully."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        print_backtest_table({})
    output = buf.getvalue()
    assert "Strategy" in output
    assert "Trades" in output


def test_print_compact_summary_empty():
    """print_compact_summary with empty summary should not crash."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        print_compact_summary({})
    # Empty summary produces no output (just the blank line)
    assert buf.getvalue() is not None


def test_print_benchmarks_formatting():
    """Benchmark output should contain expected text."""
    b1 = _mock_benchmark_result("NIFTY50 Proxy", cagr=11.2, max_dd=-14.0)
    b2 = _mock_benchmark_result("Equal Weight", cagr=9.8, max_dd=-12.1)
    b3 = _mock_benchmark_result("Buy & Hold", cagr=8.9, max_dd=-18.4)

    buf = StringIO()
    with patch("sys.stdout", buf):
        print_benchmarks(b1, b2, b3)

    output = buf.getvalue()
    assert "NIFTY50 Proxy" in output
    assert "Equal Weight" in output
    assert "Buy & Hold" in output
    assert "CAGR" in output
    assert "MaxDD" in output


def test_print_backtest_header_footer():
    """Header and footer should print without errors."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        print_backtest_header("TEST", 10, 5)
        print_backtest_footer()
    output = buf.getvalue()
    assert "UNIFIED BACKTEST" in output
    assert "TEST" in output
    assert "10 stocks" in output
    assert "Sharpe" in output


def test_walkforward_functions_no_windows():
    """Walk-forward functions should not crash with empty results."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        print_walkforward_header(10, 0, 5, 1, 1)
        print_walkforward_table([])
        print_walkforward_summary({
            "avg_oos_cagr": 0.0,
            "avg_oos_maxdd": 0.0,
            "avg_oos_trades": 0.0,
            "pct_positive_windows": 0.0,
            "wfe": "N/A",
        })
    output = buf.getvalue()
    assert "WALK-FORWARD" in output
    assert "N/A" in output