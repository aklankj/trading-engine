"""
factors/volatility.py — Volatility-based factors.

Low-volatility anomaly: lower-vol stocks tend to outperform on a risk-adjusted
basis. This factor ranks stocks by trailing 6-month realized vol so that
lower-vol stocks get higher composite scores after the base class inverts it.
"""
from __future__ import annotations
import numpy as np
from factors.base import BaseFactor


class LowVolatilityFactor(BaseFactor):
    """
    Trailing 126-day (6-month) realized daily return volatility.

    higher_is_better = False → base class inverts the z-score, so low-vol
    stocks end up with high positive z-scores in the composite.
    lookback_days = 252 → needs a full year of data to be reliable.
    """
    name = "LowVolatility"
    description = "Inverse trailing 6-month realized volatility"
    lookback_days = 252
    higher_is_better = False

    def __init__(self, vol_window: int = 126):
        self.vol_window = vol_window

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        scores = {}
        for sym, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            if as_of_date is not None:
                df = df.loc[df.index <= as_of_date]
            if len(df) < self.vol_window + 5:
                continue
            daily_rets = df["close"].pct_change().dropna()
            if len(daily_rets) < self.vol_window:
                continue
            vol = daily_rets.iloc[-self.vol_window:].std() * np.sqrt(252)
            if vol > 0:
                scores[sym] = vol
        return scores
