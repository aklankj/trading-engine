"""
factors/ — Cross-Sectional Factor Engine

The Factor Engine ranks stocks by multi-factor composite scores
and constructs a portfolio of the top-ranked stocks. Rebalanced monthly.

Key difference from strategies/:
  - strategies/ asks "should I buy THIS stock NOW?" (timing)
  - factors/ asks "which stocks have the highest expected returns?" (ranking)

Usage:
    # Backtest
    python -m factors.backtest --preset momentum_quality --universe nifty200

    # Dry-run rebalance (paper trading)
    python -m factors.rebalancer

    # Live rebalance
    python -m factors.rebalancer --live

Presets:
    momentum_quality  — 50% Momentum + 35% Quality + 15% Value (recommended)
    pure_momentum     — 100% 12-1 Momentum
    quality_first     — Quality filter (top 50%) then Momentum + Value
    vol_adjusted      — Volatility-adjusted momentum + Quality + Value
"""

from .base import (
    BaseFactor,
    FactorScore,
    CompositeScore,
    PortfolioHolding,
    RebalanceAction,
    FactorBacktestResult,
)
from .composite import (
    CompositeFactorEngine,
    momentum_quality_preset,
    pure_momentum_preset,
    quality_momentum_preset,
    vol_adjusted_preset,
    PRESETS,
)

__all__ = [
    "BaseFactor",
    "FactorScore",
    "CompositeScore",
    "PortfolioHolding",
    "RebalanceAction",
    "FactorBacktestResult",
    "CompositeFactorEngine",
    "momentum_quality_preset",
    "pure_momentum_preset",
    "quality_momentum_preset",
    "vol_adjusted_preset",
    "PRESETS",
]
