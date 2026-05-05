"""
factors/momentum.py — Cross-sectional momentum factors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from factors.base import BaseFactor


class MomentumFactor(BaseFactor):
    """12-1 month price momentum. Skip recent month to avoid reversal."""
    name = "Momentum_12_1"
    description = "12-month return excluding most recent month"
    lookback_days = 280
    higher_is_better = True

    def __init__(self, formation_days: int = 252, skip_days: int = 21):
        self.formation_days = formation_days
        self.skip_days = skip_days

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        scores = {}
        for sym, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            if as_of_date is not None:
                df = df.loc[df.index <= as_of_date]
            needed = self.formation_days + self.skip_days + 10
            if len(df) < needed:
                continue
            price_recent = df["close"].iloc[-(self.skip_days + 1)]
            price_start = df["close"].iloc[-(self.formation_days + self.skip_days)]
            if price_start <= 0:
                continue
            scores[sym] = (price_recent / price_start) - 1.0
        return scores


class AdaptiveMomentumFactor(BaseFactor):
    """
    Regime-switching momentum: 12-1 month normally, 6-1 month when the
    equal-weight universe median 3-month return drops below -15%.
    The shorter lookback catches recovery leaders faster after crashes.
    """
    name = "AdaptiveMomentum"
    description = "12-1 momentum; switches to 6-1 after market crash (median 3m ret < -15%)"
    lookback_days = 300
    higher_is_better = True

    CRASH_THRESHOLD = -0.15   # median 3m return triggering short lookback
    NORMAL_DAYS = 252          # 12-month formation
    CRASH_DAYS = 126           # 6-month formation
    SKIP_DAYS = 21             # skip most-recent month in both regimes
    REGIME_WINDOW = 63         # 3-month window for regime detection

    def _detect_regime(self, price_data, as_of_date):
        returns_3m = []
        for df in price_data.values():
            if df.empty or "close" not in df.columns:
                continue
            d = df.loc[df.index <= as_of_date] if as_of_date is not None else df
            needed = self.REGIME_WINDOW + 5
            if len(d) < needed:
                continue
            p_now = d["close"].iloc[-1]
            p_past = d["close"].iloc[-(self.REGIME_WINDOW + 1)]
            if p_past > 0:
                returns_3m.append(p_now / p_past - 1.0)
        if not returns_3m:
            return self.NORMAL_DAYS
        median_ret = float(np.median(returns_3m))
        return self.CRASH_DAYS if median_ret < self.CRASH_THRESHOLD else self.NORMAL_DAYS

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        formation_days = self._detect_regime(price_data, as_of_date)
        scores = {}
        for sym, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            if as_of_date is not None:
                df = df.loc[df.index <= as_of_date]
            needed = formation_days + self.SKIP_DAYS + 10
            if len(df) < needed:
                continue
            price_recent = df["close"].iloc[-(self.SKIP_DAYS + 1)]
            price_start = df["close"].iloc[-(formation_days + self.SKIP_DAYS)]
            if price_start <= 0:
                continue
            scores[sym] = (price_recent / price_start) - 1.0
        return scores


class VolatilityAdjustedMomentumFactor(BaseFactor):
    """Momentum / trailing volatility. Better risk-adjusted signal."""
    name = "VolAdjMomentum"
    description = "12-1 momentum / trailing 6m volatility"
    lookback_days = 300
    higher_is_better = True

    def __init__(self, formation_days=252, skip_days=21, vol_window=126):
        self.formation_days = formation_days
        self.skip_days = skip_days
        self.vol_window = vol_window

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        scores = {}
        for sym, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            if as_of_date is not None:
                df = df.loc[df.index <= as_of_date]
            needed = self.formation_days + self.skip_days + 10
            if len(df) < needed:
                continue
            price_recent = df["close"].iloc[-(self.skip_days + 1)]
            price_start = df["close"].iloc[-(self.formation_days + self.skip_days)]
            if price_start <= 0:
                continue
            momentum = (price_recent / price_start) - 1.0
            returns = df["close"].pct_change().dropna()
            if len(returns) < self.vol_window:
                continue
            vol = returns.iloc[-self.vol_window:].std() * np.sqrt(252)
            if vol < 0.01:
                continue
            scores[sym] = momentum / vol
        return scores
