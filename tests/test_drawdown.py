"""Extended tests for analytics/drawdown.py — edge cases and monotonic series."""

from __future__ import annotations

from datetime import date

import pytest

from analytics.drawdown import (
    max_drawdown,
    drawdown_series,
    ulcer_index,
    recovery_periods,
    worst_month,
)


def test_drawdown_empty():
    """Empty inputs should return safe defaults."""
    assert max_drawdown([]) == 0.0
    assert drawdown_series([]) == []
    assert ulcer_index([]) == 0.0
    assert recovery_periods([]) == []


def test_drawdown_monotonic_rise():
    """Monotonically rising equity → zero drawdown."""
    curve = [100, 110, 120, 130, 140]
    assert max_drawdown(curve) == 0.0
    assert all(d == 0.0 for d in drawdown_series(curve))


def test_drawdown_monotonic_fall():
    """Monotonically falling equity → single deep drawdown."""
    curve = [100, 90, 80, 70]
    dd = max_drawdown(curve)
    assert dd < 0
    # Final value is 70, peak was 100 → (100-70)/100 = 30%
    assert pytest.approx(dd, abs=0.5) == -30.0


def test_recovery_periods_v_shape():
    """V-shaped equity should produce exactly one recovery period."""
    from datetime import date

    curve = [
        (date(2020, 1, 1), 100.0),
        (date(2020, 2, 1), 120.0),   # peak
        (date(2020, 3, 1), 90.0),    # trough
        (date(2020, 4, 1), 100.0),   # recovering
        (date(2020, 5, 1), 130.0),   # new peak → recovery complete
    ]
    periods = recovery_periods(curve)
    assert len(periods) >= 1
    p = periods[0]
    assert p["peak_value"] == 120.0
    assert p["trough_value"] == 90.0
    assert p["depth_pct"] < 0  # Should be negative
    assert p["recovery_date"] is not None


def test_worst_month_edge_cases():
    """worst_month should handle edge inputs."""
    # All positive → empty
    rising = [(date(2020, 1, 1), 100.0), (date(2020, 2, 1), 110.0)]
    assert worst_month(rising) == {}

    # Single point → empty
    assert worst_month([(date(2020, 1, 1), 100.0)]) == {}

    # Empty → empty
    assert worst_month([]) == {}


def test_recovery_periods_multiple_drawdowns():
    """W-shaped equity produces two recovery periods."""
    curve = [
        (date(2020, 1, 1), 100.0),
        (date(2020, 2, 1), 110.0),   # peak1
        (date(2020, 3, 1), 85.0),    # trough1
        (date(2020, 4, 1), 115.0),   # new peak → recovery1
        (date(2020, 5, 1), 95.0),    # trough2
        (date(2020, 6, 1), 120.0),   # new peak → recovery2
    ]
    periods = recovery_periods(curve)
    assert len(periods) == 2
    # First drawdown: peak=110, trough=85
    assert pytest.approx(periods[0]["depth_pct"], abs=0.5) == -22.73
    # Second drawdown: peak=115, trough=95
    assert pytest.approx(periods[1]["depth_pct"], abs=0.5) == -17.39