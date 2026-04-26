"""
Macro signals — extrinsic factors that move markets.

These are structural signals most retail algos ignore:
  1. FII/DII flow momentum (NSE daily data)
  2. India VIX as fear gauge
  3. Put/Call Ratio from NIFTY options chain
  4. Global cues (US futures, crude, dollar index)
  5. Sector rotation strength
  6. Earnings momentum (post-results drift)
  7. Yield curve / bond market signals

Each signal returns a score in [-1.0, +1.0]:
  +1.0 = strongly bullish macro  |  -1.0 = strongly bearish macro
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from utils.logger import log


@dataclass
class MacroState:
    """Composite macro environment assessment."""
    overall_score: float          # -1 to +1
    fii_dii_signal: float
    vix_signal: float
    pcr_signal: float
    global_cues_signal: float
    sector_rotation: dict[str, float]
    crude_signal: float
    dollar_signal: float
    narrative: str                # Human-readable summary


# ══════════════════════════════════════════════════════════════
# INDIVIDUAL SIGNAL GENERATORS
# ══════════════════════════════════════════════════════════════

def compute_fii_dii_signal(fii_net_5d: float, dii_net_5d: float) -> float:
    """
    FII/DII flow momentum signal.

    fii_net_5d: Net FII buying in last 5 days (₹ crores, positive = buying)
    dii_net_5d: Net DII buying in last 5 days

    Logic:
      - FII buying + DII buying = strongly bullish
      - FII selling + DII buying = neutral (DII absorbing)
      - FII selling + DII selling = strongly bearish
      - FII buying + DII selling = cautiously bullish
    """
    # Normalize to roughly ±1 (typical daily flow is ±2000-5000 Cr)
    fii_norm = np.clip(fii_net_5d / 10000, -1, 1)
    dii_norm = np.clip(dii_net_5d / 10000, -1, 1)

    if fii_norm > 0.2 and dii_norm > 0.2:
        return min(1.0, (fii_norm + dii_norm) * 0.6)
    if fii_norm < -0.2 and dii_norm < -0.2:
        return max(-1.0, (fii_norm + dii_norm) * 0.6)
    if fii_norm < -0.3 and dii_norm > 0.3:
        return 0.0  # DII absorbing FII selling — neutral
    if fii_norm > 0.3 and dii_norm < -0.3:
        return fii_norm * 0.4  # FII leading, cautiously follow

    return (fii_norm * 0.6 + dii_norm * 0.4)


def compute_vix_signal(vix_current: float, vix_20d_avg: float) -> float:
    """
    VIX-based fear/greed signal.

    India VIX interpretation:
      < 12:  Extreme complacency (contrarian bearish — too calm)
      12-16: Low vol, bullish environment
      16-22: Normal
      22-30: Elevated fear, potential opportunity
      > 30:  Panic — extreme caution but also best buying opportunity

    Signal is based on VIX level AND its direction (rising vs falling).
    """
    # Level-based signal
    if vix_current < 12:
        level_signal = -0.3  # Too complacent, expect mean reversion up
    elif vix_current < 16:
        level_signal = 0.5   # Goldilocks
    elif vix_current < 22:
        level_signal = 0.0   # Neutral
    elif vix_current < 30:
        level_signal = -0.4  # Elevated fear
    else:
        level_signal = -0.8  # Panic

    # Direction: falling VIX = bullish, rising VIX = bearish
    if vix_20d_avg > 0:
        direction = (vix_20d_avg - vix_current) / vix_20d_avg
    else:
        direction = 0

    direction_signal = np.clip(direction * 2, -1, 1)

    # Combine: direction matters more than level
    return np.clip(level_signal * 0.4 + direction_signal * 0.6, -1, 1)


def compute_pcr_signal(pcr: float) -> float:
    """
    NIFTY Put/Call Ratio signal.

    PCR interpretation:
      > 1.3:  Extreme put buying — bearish sentiment (contrarian bullish)
      1.0-1.3: Moderately bearish — cautiously bullish
      0.7-1.0: Neutral
      0.5-0.7: Call heavy — bullish sentiment (contrarian bearish)
      < 0.5:  Extreme call buying — contrarian strongly bearish

    Contrarian logic: extreme sentiment tends to reverse.
    """
    if pcr > 1.5:
        return 0.8   # Extreme fear → buy
    if pcr > 1.2:
        return 0.5
    if pcr > 1.0:
        return 0.2
    if pcr > 0.8:
        return 0.0
    if pcr > 0.6:
        return -0.3
    if pcr > 0.4:
        return -0.6
    return -0.8       # Extreme greed → sell


def compute_global_cues(
    sp500_change_pct: float,
    nasdaq_change_pct: float,
    crude_change_pct: float,
    dollar_change_pct: float,
) -> tuple[float, float, float]:
    """
    Global market cues impact on Indian markets.

    Returns (global_signal, crude_signal, dollar_signal).

    Correlations to India:
      - US markets: positive correlation (~0.6 with NIFTY)
      - Crude oil: negative for India (importer) — rising crude = bearish
      - Dollar strength: negative (FII outflows when USD strengthens)
    """
    # US market impact
    us_signal = np.clip((sp500_change_pct + nasdaq_change_pct) / 2 * 0.5, -1, 1)

    # Crude: rising crude hurts India
    crude_signal = np.clip(-crude_change_pct * 0.3, -1, 1)

    # Dollar: strengthening dollar hurts emerging markets
    dollar_signal = np.clip(-dollar_change_pct * 0.4, -1, 1)

    return us_signal, crude_signal, dollar_signal


def compute_sector_rotation(sector_returns: dict[str, float]) -> dict[str, float]:
    """
    Sector rotation analysis — which sectors are leading/lagging.

    Returns normalized scores per sector.
    Classic rotation cycle:
      Early cycle: Financials, Consumer Discretionary
      Mid cycle: Technology, Industrials
      Late cycle: Energy, Materials
      Recession: Healthcare, Utilities, Consumer Staples
    """
    if not sector_returns:
        return {}

    mean_return = np.mean(list(sector_returns.values()))
    std_return = np.std(list(sector_returns.values())) or 1

    scores = {}
    for sector, ret in sector_returns.items():
        z_score = (ret - mean_return) / std_return
        scores[sector] = round(np.clip(z_score, -2, 2), 3)

    return scores


def compute_earnings_momentum(
    actual_eps: float,
    estimated_eps: float,
    price_at_results: float,
    current_price: float,
    days_since_results: int,
) -> float:
    """
    Post-earnings drift signal.

    Companies that beat estimates by >5% tend to drift higher for 20-60 days.
    Companies that miss by >5% tend to drift lower.
    This is one of the most robust anomalies in finance.
    """
    if estimated_eps <= 0:
        return 0.0

    surprise_pct = (actual_eps - estimated_eps) / abs(estimated_eps)

    if days_since_results > 60:
        return 0.0  # Signal decays after 60 days

    # Decay factor — signal strongest right after results
    decay = max(0, 1 - days_since_results / 60)

    if surprise_pct > 0.10:
        return 0.8 * decay   # Strong beat
    if surprise_pct > 0.05:
        return 0.5 * decay
    if surprise_pct < -0.10:
        return -0.8 * decay  # Strong miss
    if surprise_pct < -0.05:
        return -0.5 * decay

    return 0.0


# ══════════════════════════════════════════════════════════════
# COMPOSITE MACRO STATE
# ══════════════════════════════════════════════════════════════

def compute_macro_state(
    fii_net_5d: float = 0,
    dii_net_5d: float = 0,
    vix_current: float = 15,
    vix_20d_avg: float = 15,
    pcr: float = 0.9,
    sp500_change: float = 0,
    nasdaq_change: float = 0,
    crude_change: float = 0,
    dollar_change: float = 0,
    sector_returns: dict[str, float] = None,
) -> MacroState:
    """
    Compute composite macro state from all signals.
    Weights reflect empirical importance for Indian market:
      FII/DII:    25% (strongest predictor of short-term direction)
      VIX:        20% (fear gauge)
      Global cues: 20% (overnight US, crude, dollar)
      PCR:        15% (options market sentiment)
      Sector:     10% (rotation timing)
      Crude/Dollar: 10% (macro headwinds/tailwinds)
    """
    fii_sig = compute_fii_dii_signal(fii_net_5d, dii_net_5d)
    vix_sig = compute_vix_signal(vix_current, vix_20d_avg)
    pcr_sig = compute_pcr_signal(pcr)
    global_sig, crude_sig, dollar_sig = compute_global_cues(
        sp500_change, nasdaq_change, crude_change, dollar_change
    )
    sector_scores = compute_sector_rotation(sector_returns or {})

    # Weighted composite
    overall = (
        fii_sig * 0.25 +
        vix_sig * 0.20 +
        global_sig * 0.20 +
        pcr_sig * 0.15 +
        crude_sig * 0.10 +
        dollar_sig * 0.10
    )
    overall = np.clip(overall, -1, 1)

    # Generate narrative
    parts = []
    if fii_sig > 0.3:
        parts.append("FIIs are net buyers — institutional momentum is bullish")
    elif fii_sig < -0.3:
        parts.append("FII outflows are pressuring markets")

    if vix_sig < -0.3:
        parts.append("VIX elevated — fear in the market")
    elif vix_sig > 0.3:
        parts.append("VIX declining — calm conditions")

    if crude_sig < -0.2:
        parts.append("Rising crude is a headwind for India")
    if dollar_sig < -0.2:
        parts.append("Strong dollar driving EM outflows")

    if pcr_sig > 0.3:
        parts.append("High put buying — contrarian bullish setup")
    elif pcr_sig < -0.3:
        parts.append("Excessive call buying — caution warranted")

    narrative = ". ".join(parts) if parts else "Macro environment is neutral"

    state = MacroState(
        overall_score=round(overall, 3),
        fii_dii_signal=round(fii_sig, 3),
        vix_signal=round(vix_sig, 3),
        pcr_signal=round(pcr_sig, 3),
        global_cues_signal=round(global_sig, 3),
        sector_rotation=sector_scores,
        crude_signal=round(crude_sig, 3),
        dollar_signal=round(dollar_sig, 3),
        narrative=narrative,
    )

    log.info(f"Macro state: {overall:+.3f} | FII={fii_sig:+.2f} VIX={vix_sig:+.2f} "
             f"Global={global_sig:+.2f} PCR={pcr_sig:+.2f} | {narrative[:80]}")
    return state
