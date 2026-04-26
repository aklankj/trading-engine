"""
backtest/walkforward.py

Walk-forward validation helpers for the unified backtester.

Extracted into a separate module so tests can import these helpers
without triggering the yfinance dependency chain.

NOTE: Train windows are currently reserved periods for future
parameter calibration / optimization. Since strategies are
rule-based and not fitted, the train period simply ensures
sufficient historical context before each test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Any


def generate_walkforward_windows(
    data_index: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> list[dict[str, Any]]:
    """
    Generate non-overlapping walk-forward windows.

    Each window dict has:
        train_start, train_end, test_start, test_end  (pd.Timestamp)

    Returns empty list if data span is too short for at least one window.
    """
    if len(data_index) < 2:
        return []

    first_date = data_index[0]
    last_date = data_index[-1]
    total_days = (last_date - first_date).days
    total_years = total_days / 365.25

    if total_years < (train_years + test_years):
        return []

    windows: list[dict[str, Any]] = []
    window_num = 0

    while True:
        train_start = first_date + timedelta(days=int(window_num * step_years * 365.25))
        train_end = train_start + timedelta(days=int(train_years * 365.25))
        test_start = train_end
        test_end = test_start + timedelta(days=int(test_years * 365.25))

        if test_end > last_date:
            break

        # Snap to actual data boundaries
        train_end_snap = data_index[data_index <= train_end]
        test_start_snap = data_index[data_index >= test_start]
        test_end_snap = data_index[data_index <= test_end]

        if len(train_end_snap) == 0 or len(test_start_snap) == 0 or len(test_end_snap) == 0:
            window_num += 1
            continue

        windows.append({
            "train_start": train_start,
            "train_end": train_end_snap[-1],
            "test_start": test_start_snap[0],
            "test_end": test_end_snap[-1],
        })

        window_num += 1

    return windows


def summarize_walkforward(
    window_results: list[dict[str, Any]],
    is_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute summary metrics across all walk-forward windows.

    Parameters
    ----------
    window_results : list[dict]
        Results from _run_window (non-None entries).
    is_results : list[dict] | None
        Optional in-sample (train window) results for WFE denominator.

    Returns
    -------
    dict with avg_oos_cagr, avg_oos_maxdd, avg_oos_trades,
         pct_positive_windows, wfe
    """
    if not window_results:
        return {
            "avg_oos_cagr": 0.0,
            "avg_oos_maxdd": 0.0,
            "avg_oos_trades": 0.0,
            "pct_positive_windows": 0.0,
            "wfe": "N/A",
        }

    cagrs = [w["cagr"] for w in window_results]
    maxdds = [w["max_dd"] for w in window_results]
    trades = [w["total_trades"] for w in window_results]

    avg_oos_cagr = round(np.mean(cagrs), 2)
    avg_oos_maxdd = round(np.mean(maxdds), 2)
    avg_oos_trades = round(np.mean(trades), 1)
    pct_positive = round(sum(1 for c in cagrs if c > 0) / len(cagrs) * 100, 1)

    # WFE = Avg OOS CAGR / Avg IS CAGR
    wfe: float | str = "N/A"
    if is_results:
        is_cagrs = [r["cagr"] for r in is_results if r and r["cagr"] != 0]
        if is_cagrs:
            avg_is_cagr = np.mean(is_cagrs)
            if avg_is_cagr != 0 and avg_oos_cagr != 0:
                wfe = round(avg_oos_cagr / avg_is_cagr, 2)

    return {
        "avg_oos_cagr": avg_oos_cagr,
        "avg_oos_maxdd": avg_oos_maxdd,
        "avg_oos_trades": avg_oos_trades,
        "pct_positive_windows": pct_positive,
        "wfe": wfe,
    }