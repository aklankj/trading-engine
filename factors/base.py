"""
factors/base.py — Base classes for the Factor Engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class FactorScore:
    symbol: str
    raw_value: float
    z_score: float = 0.0
    percentile: float = 0.0
    valid: bool = True

@dataclass
class CompositeScore:
    symbol: str
    composite_z: float = 0.0
    rank: int = 0
    factor_scores: dict[str, FactorScore] = field(default_factory=dict)
    selected: bool = False

@dataclass
class PortfolioHolding:
    symbol: str
    weight: float
    shares: int = 0
    entry_price: float = 0.0
    entry_date: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    factor_rank: int = 0

@dataclass
class RebalanceAction:
    symbol: str
    action: str
    target_weight: float
    current_weight: float
    shares_delta: int
    estimated_cost: float

@dataclass
class FactorBacktestResult:
    factor_name: str
    start_date: str
    end_date: str
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_monthly_turnover: float = 0.0
    avg_holdings: int = 0
    total_rebalances: int = 0
    benchmark_cagr_pct: float = 0.0
    alpha_pct: float = 0.0
    information_ratio: float = 0.0
    hit_rate_monthly: float = 0.0
    equity_curve: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)
    yearly_returns: list[dict] = field(default_factory=list)

class BaseFactor(ABC):
    """
    Abstract base for all factors.
    Subclasses implement compute_raw() returning {symbol: raw_value}.
    Base class handles z-scoring and percentile ranking.
    """
    name: str = "BaseFactor"
    description: str = ""
    lookback_days: int = 252
    higher_is_better: bool = True

    @abstractmethod
    def compute_raw(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: dict[str, dict] | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        ...

    def score(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: dict[str, dict] | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> dict[str, FactorScore]:
        raw = self.compute_raw(price_data, fundamental_data, as_of_date)
        if len(raw) < 10:
            return {}

        symbols = list(raw.keys())
        values = np.array([raw[s] for s in symbols])

        p1, p99 = np.percentile(values, [1, 99])
        clipped = np.clip(values, p1, p99)
        mean_v, std_v = np.mean(clipped), np.std(clipped)
        z_scores = (clipped - mean_v) / std_v if std_v > 1e-10 else np.zeros_like(clipped)

        if not self.higher_is_better:
            z_scores = -z_scores

        from scipy.stats import rankdata
        ranks = rankdata(z_scores, method="average")
        n = len(ranks)
        percentiles = (ranks - 1) / max(n - 1, 1) * 100

        return {
            sym: FactorScore(
                symbol=sym,
                raw_value=round(raw[sym], 6),
                z_score=round(float(z_scores[i]), 4),
                percentile=round(float(percentiles[i]), 1),
            )
            for i, sym in enumerate(symbols)
        }
