"""
Extrinsic fundamental overlay.

Goes beyond company-level metrics to assess:
  1. Sector macro context (is the sector in a tailwind or headwind?)
  2. News/event impact detection (bad news vs structural decline)
  3. Valuation anchoring (historical PE bands, earnings yield vs bond yield)
  4. Macro regime alignment (which sectors outperform in each macro phase?)

This module adjusts the fundamental quality score up/down based on
external factors the company can't control but that affect its stock price.
"""

import numpy as np
from dataclasses import dataclass

from utils.logger import log


@dataclass
class ExtrinsicOverlay:
    """Adjustments to fundamental quality score from external factors."""
    original_score: float
    adjusted_score: float
    sector_momentum_adj: float    # -15 to +15 points
    valuation_adj: float          # -10 to +10 points
    macro_alignment_adj: float    # -10 to +10 points
    news_impact_adj: float        # -20 to +20 points
    explanation: list[str]


# ══════════════════════════════════════════════════════════════
# 1. SECTOR MACRO CONTEXT
# ══════════════════════════════════════════════════════════════

# Maps sectors to their macro sensitivity
SECTOR_MACRO_SENSITIVITY = {
    # Sector: (rate_sensitivity, crude_sensitivity, dollar_sensitivity, growth_sensitivity)
    # Positive = benefits from rising factor, negative = hurt by it
    "Banking":    (-0.6, -0.2,  0.1,  0.8),   # Hurt by rate hikes, love growth
    "NBFC":       (-0.8, -0.1,  0.0,  0.9),   # Very rate sensitive
    "IT":         ( 0.1, -0.1,  0.7,  0.5),   # Love strong dollar (exports)
    "Pharma":     ( 0.0, -0.1,  0.5,  0.3),   # Dollar benefits, defensive
    "FMCG":       ( 0.2, -0.3, -0.1,  0.3),   # Defensive, crude hurts (packaging)
    "Auto":       (-0.3, -0.5, -0.2,  0.7),   # Rate + crude sensitive
    "Consumer":   (-0.2, -0.2, -0.1,  0.6),   # Growth sensitive
    "Energy":     ( 0.0,  0.8, -0.3,  0.4),   # Love high crude (if producer)
    "Chemicals":  (-0.1, -0.4,  0.3,  0.5),   # Crude hurts (feedstock), exports
    "Infra":      (-0.4, -0.3, -0.1,  0.8),   # Rate sensitive, growth dependent
    "Telecom":    (-0.2, -0.1, -0.1,  0.4),   # Moderate sensitivity
    "Capital Markets": (-0.3, -0.1, -0.2, 0.9), # Love bull markets
    "Retail":     (-0.2, -0.3, -0.1,  0.7),   # Consumer spending dependent
    "Tech":       (-0.2, -0.1,  0.3,  0.8),   # Growth & dollar
}


def compute_sector_momentum_adj(
    sector: str,
    sector_30d_return: float,
    nifty_30d_return: float,
) -> tuple[float, str]:
    """
    Sector relative strength adjustment.

    If a sector is outperforming NIFTY by >5%, boost score.
    If underperforming by >5%, reduce score.
    If down >30% without fundamental reason, flag as potential opportunity.
    """
    relative_perf = sector_30d_return - nifty_30d_return

    if relative_perf > 0.10:
        adj = 12.0
        reason = f"{sector} leading market by {relative_perf:.0%} — strong momentum"
    elif relative_perf > 0.05:
        adj = 6.0
        reason = f"{sector} outperforming market — sector tailwind"
    elif relative_perf > -0.05:
        adj = 0.0
        reason = f"{sector} in line with market"
    elif relative_perf > -0.10:
        adj = -6.0
        reason = f"{sector} lagging market — sector headwind"
    else:
        # Down big — could be opportunity or genuine trouble
        adj = -10.0
        reason = f"{sector} significantly underperforming ({relative_perf:.0%}) — investigate cause"

        # If the sector is down >20% but historically recovers, it might be a buying opportunity
        if sector_30d_return < -0.20:
            adj = -5.0  # Less negative — could be a contrarian buy
            reason += " (potential contrarian opportunity if fundamentals intact)"

    return adj, reason


def compute_macro_alignment_adj(
    sector: str,
    rate_direction: str,     # "rising", "falling", "stable"
    crude_trend: str,        # "rising", "falling", "stable"
    dollar_trend: str,       # "strengthening", "weakening", "stable"
    growth_outlook: str,     # "expanding", "slowing", "recession"
) -> tuple[float, str]:
    """
    How aligned is this sector with the current macro regime?
    Uses sector sensitivity matrix.
    """
    sensitivities = SECTOR_MACRO_SENSITIVITY.get(sector, (0, 0, 0, 0.5))
    rate_s, crude_s, dollar_s, growth_s = sensitivities

    score = 0
    reasons = []

    # Rate impact
    rate_val = {"rising": 1, "falling": -1, "stable": 0}.get(rate_direction, 0)
    rate_impact = rate_s * rate_val
    score += rate_impact * 3
    if abs(rate_impact) > 0.3:
        direction = "benefits from" if rate_impact > 0 else "hurt by"
        reasons.append(f"{direction} {rate_direction} rates")

    # Crude impact
    crude_val = {"rising": 1, "falling": -1, "stable": 0}.get(crude_trend, 0)
    crude_impact = crude_s * crude_val
    score += crude_impact * 3
    if abs(crude_impact) > 0.3:
        direction = "benefits from" if crude_impact > 0 else "hurt by"
        reasons.append(f"{direction} {crude_trend} crude")

    # Dollar impact
    dollar_val = {"strengthening": 1, "weakening": -1, "stable": 0}.get(dollar_trend, 0)
    dollar_impact = dollar_s * dollar_val
    score += dollar_impact * 3
    if abs(dollar_impact) > 0.3:
        direction = "benefits from" if dollar_impact > 0 else "hurt by"
        reasons.append(f"{direction} {dollar_trend} dollar")

    # Growth outlook
    growth_val = {"expanding": 1, "slowing": -0.5, "recession": -1}.get(growth_outlook, 0)
    growth_impact = growth_s * growth_val
    score += growth_impact * 3
    if abs(growth_impact) > 0.3:
        direction = "benefits from" if growth_impact > 0 else "hurt by"
        reasons.append(f"{direction} {growth_outlook} growth")

    adj = np.clip(score, -10, 10)
    reason = f"Macro alignment: {'; '.join(reasons)}" if reasons else "Neutral macro alignment"

    return round(adj, 1), reason


# ══════════════════════════════════════════════════════════════
# 2. VALUATION ANCHORING
# ══════════════════════════════════════════════════════════════

def compute_valuation_adj(
    current_pe: float,
    historical_median_pe: float,
    earnings_yield: float,       # 1/PE
    bond_yield_10y: float,       # Government 10Y yield
    pe_band_low: float = None,   # 5-year PE low
    pe_band_high: float = None,  # 5-year PE high
) -> tuple[float, str]:
    """
    Valuation anchoring — is the stock cheap or expensive relative to itself
    and relative to the risk-free rate?

    Two checks:
    1. PE relative to its own 5-year range (stock-specific value)
    2. Earnings yield vs bond yield spread (opportunity cost of capital)
    """
    adj = 0.0
    reasons = []

    # ── PE relative to own history ────────────────────────────
    if historical_median_pe > 0 and current_pe > 0:
        pe_ratio = current_pe / historical_median_pe

        if pe_ratio < 0.7:
            adj += 8.0
            reasons.append(f"PE ({current_pe:.0f}) is {(1-pe_ratio):.0%} below historical median ({historical_median_pe:.0f}) — genuinely cheap for this company")
        elif pe_ratio < 0.85:
            adj += 4.0
            reasons.append(f"PE below historical median — moderate discount")
        elif pe_ratio > 1.3:
            adj -= 6.0
            reasons.append(f"PE ({current_pe:.0f}) is {(pe_ratio-1):.0%} above median ({historical_median_pe:.0f}) — expensive")
        elif pe_ratio > 1.15:
            adj -= 3.0
            reasons.append(f"PE slightly above historical median")

    # ── PE band extremes ──────────────────────────────────────
    if pe_band_low and pe_band_high and current_pe > 0:
        pe_range = pe_band_high - pe_band_low
        if pe_range > 0:
            pe_position = (current_pe - pe_band_low) / pe_range
            if pe_position < 0.2:
                adj += 5.0
                reasons.append(f"Near bottom of 5Y PE range ({pe_band_low:.0f}-{pe_band_high:.0f})")
            elif pe_position > 0.8:
                adj -= 4.0
                reasons.append(f"Near top of 5Y PE range")

    # ── Earnings yield vs bond yield ──────────────────────────
    if earnings_yield > 0 and bond_yield_10y > 0:
        spread = earnings_yield - bond_yield_10y

        if spread > 0.03:
            adj += 5.0
            reasons.append(f"Earnings yield ({earnings_yield:.1%}) well above bond yield ({bond_yield_10y:.1%}) — equities attractive")
        elif spread > 0.01:
            adj += 2.0
            reasons.append(f"Earnings yield moderately above bonds")
        elif spread < -0.02:
            adj -= 5.0
            reasons.append(f"Bonds yielding more than earnings — stock expensive relative to fixed income")

    adj = np.clip(adj, -10, 10)
    reason = "; ".join(reasons) if reasons else "Valuation in line with history"
    return round(adj, 1), reason


# ══════════════════════════════════════════════════════════════
# 3. NEWS/EVENT IMPACT ASSESSMENT
# ══════════════════════════════════════════════════════════════

def compute_news_impact_adj(
    news_sentiment: float,       # -1 to +1 from FinBERT/LLM analysis
    is_temporary: bool,          # Temporary event vs structural change
    drop_magnitude_pct: float,   # How much the stock dropped from the news
) -> tuple[float, str]:
    """
    Assess whether a price drop is a buying opportunity or a genuine red flag.

    Key distinction:
    - Temporary bad news (regulatory fine, one-time expense, sector rotation)
      → If fundamentals intact, this is a BUY opportunity
    - Structural bad news (market share loss, technology disruption, governance)
      → This is a genuine deterioration, DO NOT buy the dip

    This requires human judgment (or LLM analysis). The is_temporary flag
    should be set by the research scanner / Claude analysis.
    """
    if abs(news_sentiment) < 0.2:
        return 0.0, "No significant news impact"

    if news_sentiment < -0.3:
        if is_temporary and drop_magnitude_pct > 15:
            # Bad news + temporary + big drop = potential opportunity
            adj = min(15.0, drop_magnitude_pct * 0.5)
            reason = (f"Temporary negative news drove {drop_magnitude_pct:.0f}% drop — "
                      f"if fundamentals intact, this is a buying opportunity")
        elif is_temporary:
            adj = 3.0
            reason = "Minor temporary headwind — fundamentals likely unaffected"
        else:
            # Structural bad news
            adj = max(-20.0, news_sentiment * 20)
            reason = (f"Structural negative development — "
                      f"reassess fundamental thesis before buying")
    elif news_sentiment > 0.3:
        adj = min(10.0, news_sentiment * 10)
        reason = "Positive development — fundamentals may be improving"
    else:
        adj = 0.0
        reason = "Mixed news — no clear impact"

    return round(adj, 1), reason


# ══════════════════════════════════════════════════════════════
# COMPOSITE EXTRINSIC OVERLAY
# ══════════════════════════════════════════════════════════════

def apply_extrinsic_overlay(
    base_score: float,
    sector: str,
    sector_30d_return: float = 0,
    nifty_30d_return: float = 0,
    rate_direction: str = "stable",
    crude_trend: str = "stable",
    dollar_trend: str = "stable",
    growth_outlook: str = "expanding",
    current_pe: float = 0,
    historical_median_pe: float = 0,
    earnings_yield: float = 0,
    bond_yield_10y: float = 0.07,
    pe_band_low: float = None,
    pe_band_high: float = None,
    news_sentiment: float = 0,
    is_temporary_news: bool = True,
    drop_magnitude_pct: float = 0,
) -> ExtrinsicOverlay:
    """
    Apply all extrinsic adjustments to a base quality score.
    The adjusted score determines buy/sell decisions for fundamental picks.
    """
    explanations = []

    # Sector momentum
    sector_adj, sector_reason = compute_sector_momentum_adj(
        sector, sector_30d_return, nifty_30d_return
    )
    explanations.append(sector_reason)

    # Macro alignment
    macro_adj, macro_reason = compute_macro_alignment_adj(
        sector, rate_direction, crude_trend, dollar_trend, growth_outlook
    )
    explanations.append(macro_reason)

    # Valuation
    val_adj, val_reason = compute_valuation_adj(
        current_pe, historical_median_pe, earnings_yield,
        bond_yield_10y, pe_band_low, pe_band_high
    )
    explanations.append(val_reason)

    # News impact
    news_adj, news_reason = compute_news_impact_adj(
        news_sentiment, is_temporary_news, drop_magnitude_pct
    )
    if abs(news_adj) > 0:
        explanations.append(news_reason)

    # Compute adjusted score (clamped to 0-100)
    adjusted = base_score + sector_adj + macro_adj + val_adj + news_adj
    adjusted = np.clip(adjusted, 0, 100)

    result = ExtrinsicOverlay(
        original_score=base_score,
        adjusted_score=round(adjusted, 1),
        sector_momentum_adj=sector_adj,
        valuation_adj=val_adj,
        macro_alignment_adj=macro_adj,
        news_impact_adj=news_adj,
        explanation=explanations,
    )

    log.info(
        f"Extrinsic overlay: {base_score:.0f} → {adjusted:.0f} | "
        f"Sector={sector_adj:+.0f} Macro={macro_adj:+.0f} "
        f"Val={val_adj:+.0f} News={news_adj:+.0f}"
    )
    return result
