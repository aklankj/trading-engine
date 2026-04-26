"""
Market regime detection.

Detects 5 regimes: Bull, Bear, Sideways, HighVol, Recovery
Uses a combination of:
  1. Rolling return/volatility statistics (fast, always available)
  2. Optional HMM for more sophisticated state inference

The regime drives the meta-allocator's strategy weights.
"""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass

from utils.logger import log

RegimeType = Literal["Bull", "Bear", "Sideways", "HighVol", "Recovery", "Unknown"]


@dataclass
class RegimeState:
    """Current regime with confidence and supporting metrics."""
    regime: RegimeType
    confidence: float          # 0.0 - 1.0
    ann_return: float          # Annualized return (rolling)
    ann_volatility: float      # Annualized volatility (rolling)
    trend_strength: float      # |mean_return| / volatility
    momentum_20d: float        # 20-day price momentum
    momentum_50d: float        # 50-day price momentum
    vol_regime: str            # "low", "normal", "high", "extreme"
    smoothed_signal: float     # -1.0 (deep bear) to +1.0 (strong bull)


def detect_regime(
    df: pd.DataFrame,
    lookback: int = 60,
    vol_lookback: int = 20,
) -> RegimeState:
    """
    Detect current market regime from OHLCV DataFrame.

    Args:
        df: DataFrame with 'close' column (and optionally computed indicators)
        lookback: Days for trend regime calculation
        vol_lookback: Days for volatility regime calculation

    Returns:
        RegimeState with regime classification and metrics
    """
    if len(df) < lookback + 10:
        return RegimeState(
            regime="Unknown", confidence=0.0, ann_return=0.0,
            ann_volatility=0.0, trend_strength=0.0,
            momentum_20d=0.0, momentum_50d=0.0,
            vol_regime="unknown", smoothed_signal=0.0,
        )

    close = df["close"].values
    returns = np.diff(close) / close[:-1]

    # ── Rolling statistics (last N days) ──────────────────────
    recent_returns = returns[-lookback:]
    mean_ret = np.mean(recent_returns)
    vol = np.std(recent_returns)
    ann_ret = mean_ret * 252
    ann_vol = vol * np.sqrt(252)
    trend_strength = abs(mean_ret) / (vol + 1e-10)

    # ── Short-term volatility regime ──────────────────────────
    short_vol = np.std(returns[-vol_lookback:]) * np.sqrt(252)

    if short_vol > 0.40:
        vol_regime = "extreme"
    elif short_vol > 0.25:
        vol_regime = "high"
    elif short_vol > 0.12:
        vol_regime = "normal"
    else:
        vol_regime = "low"

    # ── Momentum signals ──────────────────────────────────────
    mom_20 = (close[-1] / close[-min(21, len(close))] - 1) if len(close) > 20 else 0
    mom_50 = (close[-1] / close[-min(51, len(close))] - 1) if len(close) > 50 else 0

    # ── Smoothed directional signal ───────────────────────────
    # Combines multiple timeframes into -1 to +1 score
    signals = []
    for period in [10, 20, 40, 60]:
        if len(returns) >= period:
            p_ret = np.mean(returns[-period:]) * 252
            signals.append(np.clip(p_ret / 0.20, -1, 1))  # Normalize to ±1
    smoothed = np.mean(signals) if signals else 0.0

    # ── Regime classification ─────────────────────────────────
    regime, confidence = _classify_regime(
        ann_ret, ann_vol, trend_strength, mom_20, mom_50, vol_regime, smoothed
    )

    state = RegimeState(
        regime=regime,
        confidence=confidence,
        ann_return=ann_ret,
        ann_volatility=ann_vol,
        trend_strength=trend_strength,
        momentum_20d=mom_20,
        momentum_50d=mom_50,
        vol_regime=vol_regime,
        smoothed_signal=smoothed,
    )

    log.debug(
        f"Regime: {regime} (conf={confidence:.2f}) | "
        f"AnnRet={ann_ret:.1%} AnnVol={ann_vol:.1%} | "
        f"Mom20={mom_20:.2%} Mom50={mom_50:.2%} | VolRegime={vol_regime}"
    )
    return state


def _classify_regime(
    ann_ret: float,
    ann_vol: float,
    trend_strength: float,
    mom_20: float,
    mom_50: float,
    vol_regime: str,
    smoothed: float,
) -> tuple[RegimeType, float]:
    """
    Rule-based regime classifier with confidence scoring.
    Returns (regime, confidence).

    Priority order:
    1. Extreme volatility overrides everything → HighVol
    2. Strong directional moves → Bull or Bear
    3. Recovery pattern (recent upturn after decline)
    4. Default → Sideways
    """

    # ── HighVol: extreme volatility regardless of direction ───
    if vol_regime == "extreme" or (vol_regime == "high" and abs(ann_ret) < 0.10):
        conf = 0.9 if vol_regime == "extreme" else 0.7
        return "HighVol", conf

    # ── Bull: strong uptrend ─────────────────────────────────
    if ann_ret > 0.12 and trend_strength > 0.05 and mom_20 > 0.02:
        conf = min(0.95, 0.6 + trend_strength * 2)
        return "Bull", conf

    if ann_ret > 0.05 and mom_20 > 0.03 and mom_50 > 0.05:
        return "Bull", 0.65

    # ── Bear: strong downtrend ───────────────────────────────
    if ann_ret < -0.12 and trend_strength > 0.05 and mom_20 < -0.02:
        conf = min(0.95, 0.6 + trend_strength * 2)
        return "Bear", conf

    if ann_ret < -0.05 and mom_20 < -0.03 and mom_50 < -0.05:
        return "Bear", 0.65

    # ── Recovery: short-term up after medium-term down ────────
    if mom_20 > 0.03 and mom_50 < 0.0 and ann_vol > 0.15:
        return "Recovery", 0.6

    if mom_20 > 0.02 and ann_ret < 0.0 and smoothed > 0.1:
        return "Recovery", 0.55

    # ── Sideways: low directional conviction ─────────────────
    if abs(ann_ret) < 0.08 and trend_strength < 0.04:
        conf = 0.7 if vol_regime == "low" else 0.55
        return "Sideways", conf

    # ── Fallback based on smoothed signal ─────────────────────
    if smoothed > 0.3:
        return "Bull", 0.50
    if smoothed < -0.3:
        return "Bear", 0.50

    return "Sideways", 0.45


def detect_regime_hmm(df: pd.DataFrame, n_states: int = 3) -> RegimeState:
    """
    HMM-based regime detection using hmmlearn.
    More sophisticated but requires more data (200+ days recommended).

    Falls back to rule-based if hmmlearn is unavailable or data is insufficient.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.warning("hmmlearn not installed, falling back to rule-based regime detection")
        return detect_regime(df)

    if len(df) < 200:
        log.debug("Insufficient data for HMM, using rule-based detection")
        return detect_regime(df)

    returns = df["close"].pct_change().dropna().values.reshape(-1, 1)

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(returns)
        states = model.predict(returns)
        current_state = states[-1]

        # Map HMM states to regime names based on mean returns
        state_means = model.means_.flatten()
        state_vols = np.sqrt(model.covars_.flatten())
        sorted_indices = np.argsort(state_means)

        regime_map = {}
        if n_states == 3:
            regime_map[sorted_indices[0]] = "Bear"
            regime_map[sorted_indices[1]] = "Sideways"
            regime_map[sorted_indices[2]] = "Bull"

        hmm_regime = regime_map.get(current_state, "Unknown")

        # Get the rule-based state for metrics
        rule_state = detect_regime(df)

        # Override regime with HMM result but keep metrics
        smoothed_probs = model.predict_proba(returns)[-1]
        confidence = float(smoothed_probs[current_state])

        # Check for high-vol override
        if rule_state.vol_regime in ("extreme", "high") and rule_state.ann_volatility > 0.35:
            hmm_regime = "HighVol"

        # Check for recovery pattern
        if hmm_regime == "Bull" and rule_state.momentum_50d < -0.03:
            hmm_regime = "Recovery"

        return RegimeState(
            regime=hmm_regime,
            confidence=confidence,
            ann_return=rule_state.ann_return,
            ann_volatility=rule_state.ann_volatility,
            trend_strength=rule_state.trend_strength,
            momentum_20d=rule_state.momentum_20d,
            momentum_50d=rule_state.momentum_50d,
            vol_regime=rule_state.vol_regime,
            smoothed_signal=rule_state.smoothed_signal,
        )

    except Exception as e:
        log.warning(f"HMM fitting failed: {e}, falling back to rule-based")
        return detect_regime(df)
