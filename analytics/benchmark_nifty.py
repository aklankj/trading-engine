"""
analytics/benchmark_nifty.py

Real NIFTY 50 benchmark using ^NSEI index data from yfinance.

Computes calendar-time CAGR, peak-to-trough max drawdown, and equity curve.
Returns NaN for metrics on any failure (no crash).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:  # noqa: BLE001 — yfinance may crash on Python 3.9 due to type-syntax incompatibility
    HAS_YFINANCE = False

import numpy as np
import pandas as pd


# ──────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────


@dataclass
class NiftyBenchmark:
    """Real NIFTY 50 benchmark result.

    Attributes are float('nan') / empty list on failure so callers
    can always unpack without guarding types.
    """

    cagr: float = float("nan")
    max_drawdown: float = float("nan")
    equity_curve: list[tuple[date, float]] = field(default_factory=list)


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def _to_timestamp(d: str | date | pd.Timestamp | None) -> Optional[pd.Timestamp]:
    """Normalize date-like input to pd.Timestamp or None."""
    if d is None:
        return None
    return pd.to_datetime(d)


# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────


def compute_nifty_benchmark(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> NiftyBenchmark:
    """
    Fetch real NIFTY 50 index (^NSEI) and compute benchmark metrics.

    Parameters
    ----------
    start_date, end_date :
        Date range for the benchmark. Calendar-time CAGR is used
        (not trading-day count).

    Returns
    -------
    NiftyBenchmark
        CAGR, max drawdown, and full equity curve. On any failure
        (no yfinance, download error, empty data) returns a result
        with NaN / empty curve.
    """
    if not HAS_YFINANCE:
        return NiftyBenchmark()

    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date)
    if start is None or end is None:
        return NiftyBenchmark()

    try:
        raw = yf.download(
            "^NSEI",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception:  # noqa: BLE001
        return NiftyBenchmark()

    if raw.empty or "Close" not in raw.columns:
        return NiftyBenchmark()

    # auto_adjust=True → Close is already adjusted
    close_prices = raw["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]  # flatten multi-level columns

    prices = close_prices.dropna().values.astype(np.float64)
    if len(prices) < 2:
        return NiftyBenchmark()

    # Equity curve
    equity_curve: list[tuple[date, float]] = []
    for idx, val in zip(close_prices.dropna().index, prices):
        d = idx.date() if hasattr(idx, "date") else idx
        equity_curve.append((d, float(val)))

    # CAGR — calendar time
    first_date = close_prices.dropna().index[0]
    last_date = close_prices.dropna().index[-1]
    years = (last_date - first_date).days / 365.25
    if years <= 0.01:
        return NiftyBenchmark()

    total_return = prices[-1] / prices[0]
    cagr = float((total_return ** (1.0 / years)) - 1.0)

    # Max drawdown — peak to trough on prices
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    max_dd = float(dd.min())

    return NiftyBenchmark(
        cagr=round(cagr * 100, 1),  # return as percentage
        max_drawdown=round(max_dd * 100, 1),
        equity_curve=equity_curve,
    )