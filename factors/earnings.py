"""
factors/earnings.py — Post-Earnings Announcement Drift (PEAD) factor.

Theory: After an earnings surprise, stocks drift in the surprise direction for
30-60 days as the market slowly incorporates the new information.

Two paths, used in priority order:
  1. Fundamental acceleration (if Screener.in has profit_growth_1 Year AND
     profit_growth_3 Years for ≥10 stocks): measures how much recent growth
     exceeds trailing average.
  2. Price-based proxy (fallback): measures the abnormal 1-month return
     standardized by trailing volatility — a positive abnormal return is the
     best available signal for an earnings beat when quarterly data is absent.

The two paths produce equivalent z-scores (the base class normalizes both),
so switching between them as data improves is seamless.
"""
from __future__ import annotations
import numpy as np
from factors.base import BaseFactor


class EarningsMomentumFactor(BaseFactor):
    """
    PEAD factor: earnings acceleration or price-proxy for abnormal return.

    Raw value:
      Fundamental: (pg_1yr - pg_trailing) / max(|pg_trailing|, 1)
      Price proxy: ret_1m / expected_1m_vol  (standardized abnormal return)

    higher_is_better = True — stocks with positive earnings surprises are favored.
    lookback_days = 63 — one quarter; stale beyond that.
    """
    name = "EarningsMomentum"
    description = "Earnings acceleration (fundamental) or abnormal 1m return (price proxy)"
    lookback_days = 63
    higher_is_better = True

    # Price-proxy windows
    WINDOW_SHORT = 22   # ~1 calendar month
    WINDOW_VOL = 126    # 6-month vol baseline

    # Minimum stocks for the fundamental path to be considered valid
    FUND_MIN_STOCKS = 10

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        if fundamental_data:
            fund_scores = self._fundamental_scores(fundamental_data)
            if len(fund_scores) >= self.FUND_MIN_STOCKS:
                return fund_scores
        return self._price_proxy_scores(price_data, as_of_date)

    # ── Fundamental path ────────────────────────────────────────────────────

    def _fundamental_scores(self, fundamental_data: dict) -> dict[str, float]:
        """
        Earnings acceleration = how much 1-year growth exceeds trailing average.

        Uses profit_growth_1 Year (or TTM) vs profit_growth_3 Years (or 5yr).
        If 1yr >> trailing → stock is in earnings acceleration → positive signal.
        """
        scores = {}
        for sym, fund in fundamental_data.items():
            pg_1yr = fund.get("profit_growth_1 Year",
                              fund.get("profit_growth_TTM",
                              fund.get("profit_growth_1Years", 0)))
            pg_3yr = fund.get("profit_growth_3 Years",
                              fund.get("profit_growth_3Years", 0))
            pg_5yr = fund.get("profit_growth_5 Years",
                              fund.get("profit_growth_5Years", 0))

            # Need at least recent and one trailing figure, both non-zero
            trailing = pg_3yr if pg_3yr != 0 else pg_5yr
            if pg_1yr == 0 or trailing == 0:
                continue

            # Acceleration: excess return over trailing, normalized
            raw = (pg_1yr - trailing) / max(abs(trailing), 1.0)
            scores[sym] = raw
        return scores

    # ── Price-proxy path ────────────────────────────────────────────────────

    def _price_proxy_scores(self, price_data: dict, as_of_date) -> dict[str, float]:
        """
        Abnormal 1-month return = ret_1m / expected_vol_1m.

        expected_vol_1m = trailing_daily_vol * sqrt(WINDOW_SHORT)

        A large positive abnormal return indicates an earnings beat that the
        market is still absorbing → PEAD continuation expected.
        """
        scores = {}
        for sym, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            if as_of_date is not None:
                df = df.loc[df.index <= as_of_date]
            needed = self.WINDOW_VOL + self.WINDOW_SHORT + 5
            if len(df) < needed:
                continue

            # 1-month raw return (no skip — we want the recent surprise window)
            ret_1m = df["close"].iloc[-1] / df["close"].iloc[-(self.WINDOW_SHORT + 1)] - 1.0

            # 6-month daily return vol, computed on pre-surprise window to avoid
            # contaminating the baseline with the earnings move itself
            baseline = df["close"].iloc[-(self.WINDOW_VOL + self.WINDOW_SHORT):
                                        -self.WINDOW_SHORT]
            daily_rets = baseline.pct_change().dropna()
            if len(daily_rets) < 30:
                continue
            vol_daily = daily_rets.std()
            if vol_daily < 1e-6:
                continue

            expected_vol_1m = vol_daily * np.sqrt(self.WINDOW_SHORT)
            scores[sym] = ret_1m / expected_vol_1m
        return scores
