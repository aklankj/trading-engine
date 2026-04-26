"""
analytics/drawdown.py

Drawdown and risk metrics for equity curves.

All functions are pure — no state, no side effects.
Designed to work with both:
  - backtest/unified.py  (equity curve stored as list[float])
  - portfolio/simulator.py (equity curve stored as list[tuple[date, float]])
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any


# ──────────────────────────────────────────
# Value-only functions (input: list[float])
# ──────────────────────────────────────────


def max_drawdown(equity_curve: list[float]) -> float:
    """
    Maximum peak-to-trough drawdown as a negative percentage.

    Example
    -------
    >>> max_drawdown([100, 110, 90, 95, 80, 100])
    -27.27
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
    return round(-max_dd, 2)


def drawdown_series(equity_curve: list[float]) -> list[float]:
    """
    Running drawdown at each point in the equity curve.
    Returns negative percentages (e.g. -5.3 means 5.3% below peak).
    """
    if not equity_curve:
        return []

    peak = equity_curve[0]
    series: list[float] = []
    for e in equity_curve:
        peak = max(peak, e)
        dd = 0.0
        if peak > 0:
            dd = (peak - e) / peak * 100
        series.append(round(-dd, 2))
    return series


def ulcer_index(equity_curve: list[float]) -> float:
    """
    Ulcer Index — measures both depth and duration of drawdowns.

    UI = sqrt(mean(D_i²))
    where D_i is the i-th period drawdown percentage (as positive value).

    Lower is better. Zero means no drawdown.
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    squared_dds: list[float] = []
    for e in equity_curve:
        peak = max(peak, e)
        dd = 0.0
        if peak > 0:
            dd = (peak - e) / peak * 100  # as positive percentage
        squared_dds.append(dd * dd)

    mean_sq = sum(squared_dds) / len(squared_dds)
    return round(math.sqrt(mean_sq), 2)


# ──────────────────────────────────────────
# Date-aware functions (input: list[tuple[date, float]])
# ──────────────────────────────────────────


def recovery_periods(
    equity_curve: list[tuple[date, float]],
) -> list[dict[str, Any]]:
    """
    Identify each peak-to-trough-to-recovery cycle.

    Returns a list of dicts, one per recovery cycle:
        {
            "peak_date": date,
            "peak_value": float,
            "trough_date": date,
            "trough_value": float,
            "depth_pct": float,       # negative drawdown %
            "recovery_date": date | None,  # None if not yet recovered
            "recovery_value": float | None,
            "duration_days": int,      # peak → trough
            "recovery_days": int | None,  # trough → recovery
        }
    """
    if not equity_curve:
        return []

    periods: list[dict[str, Any]] = []
    peak_date = equity_curve[0][0]
    peak_value = equity_curve[0][1]
    trough_date = peak_date
    trough_value = peak_value
    in_drawdown = False

    for d, val in equity_curve:
        if val > peak_value:
            # New high — if we were in drawdown, mark recovery
            if in_drawdown:
                periods.append({
                    "peak_date": peak_date,
                    "peak_value": peak_value,
                    "trough_date": trough_date,
                    "trough_value": trough_value,
                    "depth_pct": round(-(peak_value - trough_value) / peak_value * 100, 2),
                    "recovery_date": d,
                    "recovery_value": val,
                    "duration_days": (trough_date - peak_date).days,
                    "recovery_days": (d - trough_date).days,
                })
                in_drawdown = False
            peak_date = d
            peak_value = val
            trough_date = d
            trough_value = val
        elif val < trough_value:
            # New trough
            trough_date = d
            trough_value = val
            in_drawdown = True
        # else: recovery in progress but not yet at new peak — do nothing

    # Handle unfinished drawdown at end of series
    if in_drawdown:
        periods.append({
            "peak_date": peak_date,
            "peak_value": peak_value,
            "trough_date": trough_date,
            "trough_value": trough_value,
            "depth_pct": round(-(peak_value - trough_value) / peak_value * 100, 2),
            "recovery_date": None,
            "recovery_value": None,
            "duration_days": (trough_date - peak_date).days,
            "recovery_days": None,
        })

    return periods


def worst_month(equity_curve: list[tuple[date, float]]) -> dict[str, Any]:
    """
    Find the calendar month with the worst return.

    Groups equity values by (year, month) and computes the return
    from the first to last equity value within each month.

    Returns
    -------
    dict with keys: "month" (str, "YYYY-MM"), "return_pct" (float, negative)
    or empty dict if input has fewer than 2 data points.
    """
    if len(equity_curve) < 2:
        return {}

    # Group by (year, month)
    monthly_first: dict[tuple[int, int], float] = {}
    monthly_last: dict[tuple[int, int], float] = {}

    for d, val in equity_curve:
        key = (d.year, d.month)
        if key not in monthly_first:
            monthly_first[key] = val
        monthly_last[key] = val

    worst = None
    worst_ret = 0.0

    for key in monthly_first:
        first_val = monthly_first[key]
        last_val = monthly_last[key]
        if first_val > 0:
            ret = (last_val - first_val) / first_val * 100
            # Only track negative returns
            if ret < worst_ret:
                worst_ret = ret
                worst = key

    if worst is None:
        return {}

    return {
        "month": f"{worst[0]}-{worst[1]:02d}",
        "return_pct": round(worst_ret, 2),
    }