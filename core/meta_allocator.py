"""
Meta-allocator — the brain that weighs strategies by regime.

Takes individual strategy signals and the current regime,
applies regime-specific weights, and produces a single
composite signal that the risk gate evaluates.
"""

import numpy as np
from dataclasses import dataclass

from core.regime import RegimeType, RegimeState
from core.strategies import STRATEGIES, run_all_strategies
from utils.logger import log

import pandas as pd


# ── Regime → Strategy weight mapping ─────────────────────────
# Each regime distributes 100% across the 8 strategies
# Weights should sum to 1.0 per regime

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "Bull": {
        "dual_ma": 0.25, "macd_momentum": 0.20, "kaufman_adaptive": 0.15,
        "donchian_breakout": 0.15, "mtf_consensus": 0.10,
        "rsi_reversion": 0.05, "bollinger_squeeze": 0.05, "vol_risk_parity": 0.05,
    },
    "Bear": {
        "kaufman_adaptive": 0.20, "vol_risk_parity": 0.20, "dual_ma": 0.15,
        "mtf_consensus": 0.15, "rsi_reversion": 0.10, "macd_momentum": 0.10,
        "bollinger_squeeze": 0.05, "donchian_breakout": 0.05,
    },
    "Sideways": {
        "rsi_reversion": 0.25, "bollinger_squeeze": 0.20, "kaufman_adaptive": 0.15,
        "vol_risk_parity": 0.15, "mtf_consensus": 0.10,
        "dual_ma": 0.05, "macd_momentum": 0.05, "donchian_breakout": 0.05,
    },
    "HighVol": {
        "vol_risk_parity": 0.30, "bollinger_squeeze": 0.20, "kaufman_adaptive": 0.15,
        "rsi_reversion": 0.10, "donchian_breakout": 0.10, "mtf_consensus": 0.10,
        "dual_ma": 0.025, "macd_momentum": 0.025,
    },
    "Recovery": {
        "macd_momentum": 0.20, "rsi_reversion": 0.20, "vol_risk_parity": 0.15,
        "mtf_consensus": 0.15, "kaufman_adaptive": 0.10, "dual_ma": 0.10,
        "donchian_breakout": 0.05, "bollinger_squeeze": 0.05,
    },
    "Unknown": {
        "kaufman_adaptive": 0.20, "vol_risk_parity": 0.20, "mtf_consensus": 0.15,
        "rsi_reversion": 0.10, "bollinger_squeeze": 0.10,
        "dual_ma": 0.10, "macd_momentum": 0.10, "donchian_breakout": 0.05,
    },
}


@dataclass
class CompositeSignal:
    """Output of the meta-allocator."""
    signal: float                      # -1.0 to +1.0
    direction: str                     # "BUY", "SELL", "HOLD"
    strength: str                      # "strong", "moderate", "weak"
    regime: RegimeState                # Current regime info
    strategy_signals: dict[str, float] # Individual strategy signals
    strategy_weights: dict[str, float] # Applied weights for this regime
    weighted_breakdown: dict[str, float]  # signal * weight per strategy
    agreement_pct: float               # % of strategies agreeing with direction


def compute_composite(
    df: pd.DataFrame,
    regime: RegimeState,
    signal_threshold: float = 0.3,
) -> CompositeSignal:
    """
    Run all strategies, apply regime-weighted combination, return composite signal.

    Args:
        df: OHLCV DataFrame with sufficient history
        regime: Current RegimeState from detector
        signal_threshold: Minimum absolute signal to act on (below = HOLD)
    """
    # Run all strategies
    signals = run_all_strategies(df)

    # Get weights for current regime
    weights = REGIME_WEIGHTS.get(regime.regime, REGIME_WEIGHTS["Unknown"])

    # Compute weighted composite
    composite = 0.0
    breakdown = {}
    for key, signal in signals.items():
        w = weights.get(key, 0.0)
        weighted = signal * w
        breakdown[key] = weighted
        composite += weighted

    # Clamp to [-1, 1]
    composite = float(np.clip(composite, -1.0, 1.0))

    # Scale by regime confidence (less confident → smaller signals)
    composite *= (0.5 + 0.5 * regime.confidence)

    # Direction
    if composite > signal_threshold:
        direction = "BUY"
    elif composite < -signal_threshold:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Strength
    abs_sig = abs(composite)
    if abs_sig > 0.7:
        strength = "strong"
    elif abs_sig > 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    # Agreement: % of strategies pointing same direction as composite
    if composite != 0:
        sign = np.sign(composite)
        agreeing = sum(1 for s in signals.values() if np.sign(s) == sign and abs(s) > 0.1)
        agreement = agreeing / max(len(signals), 1)
    else:
        agreement = 0.0

    result = CompositeSignal(
        signal=composite,
        direction=direction,
        strength=strength,
        regime=regime,
        strategy_signals=signals,
        strategy_weights=weights,
        weighted_breakdown=breakdown,
        agreement_pct=agreement,
    )

    log.info(
        f"Composite: {composite:+.3f} ({direction} {strength}) | "
        f"Regime: {regime.regime} | Agreement: {agreement:.0%} | "
        f"Top contributors: {_top_contributors(breakdown)}"
    )
    return result


def _top_contributors(breakdown: dict[str, float], n: int = 3) -> str:
    """Format the top N contributing strategies."""
    sorted_items = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
    parts = []
    for key, val in sorted_items[:n]:
        name = STRATEGIES[key].name
        parts.append(f"{name}={val:+.3f}")
    return ", ".join(parts)
