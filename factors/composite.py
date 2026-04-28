"""
factors/composite.py — Multi-factor ranking engine.

Combines factors → ranks stocks → selects top N → assigns weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from factors.base import BaseFactor, FactorScore, CompositeScore, PortfolioHolding
from factors.momentum import MomentumFactor, VolatilityAdjustedMomentumFactor
from factors.quality import QualityFactor
from factors.value import ValueFactor


@dataclass
class FactorConfig:
    factor: BaseFactor
    weight: float
    min_percentile: float = 0.0  # Hard filter: must be above this percentile


def momentum_quality_preset():
    """50% Momentum + 35% Quality + 15% Value (recommended)."""
    return [
        FactorConfig(factor=MomentumFactor(), weight=0.50),
        FactorConfig(factor=QualityFactor(), weight=0.35),
        FactorConfig(factor=ValueFactor(), weight=0.15),
    ]

def pure_momentum_preset():
    """100% 12-1 momentum."""
    return [FactorConfig(factor=MomentumFactor(), weight=1.0)]

def quality_momentum_preset():
    """Quality filter (top 50%) then Momentum + Value."""
    return [
        FactorConfig(factor=QualityFactor(), weight=0.0, min_percentile=50.0),
        FactorConfig(factor=MomentumFactor(), weight=0.70),
        FactorConfig(factor=ValueFactor(), weight=0.30),
    ]

def vol_adjusted_preset():
    """Vol-adjusted momentum + Quality + Value."""
    return [
        FactorConfig(factor=VolatilityAdjustedMomentumFactor(), weight=0.50),
        FactorConfig(factor=QualityFactor(), weight=0.35),
        FactorConfig(factor=ValueFactor(), weight=0.15),
    ]

PRESETS = {
    "momentum_quality": momentum_quality_preset,
    "pure_momentum": pure_momentum_preset,
    "quality_first": quality_momentum_preset,
    "vol_adjusted": vol_adjusted_preset,
}


class CompositeFactorEngine:
    def __init__(
        self,
        factors: list[FactorConfig] | None = None,
        top_n: int = 20,
        min_market_cap: float = 1000,
        min_avg_volume: float = 500_000,
        max_sector_pct: float = 0.30,
        weighting: str = "equal",
    ):
        self.factors = factors or momentum_quality_preset()
        self.top_n = top_n
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        self.max_sector_pct = max_sector_pct
        self.weighting = weighting

    def rank_universe(self, price_data, fundamental_data=None, as_of_date=None):
        all_scores = {}
        eligible = None

        for fc in self.factors:
            scores = fc.factor.score(price_data, fundamental_data, as_of_date)
            if not scores:
                continue
            all_scores[fc.factor.name] = scores
            if fc.min_percentile > 0:
                passing = {s for s, sc in scores.items() if sc.percentile >= fc.min_percentile}
                eligible = passing if eligible is None else eligible & passing

        if not all_scores:
            return []

        common = None
        for scores in all_scores.values():
            syms = set(scores.keys())
            common = syms if common is None else common & syms
        if eligible is not None:
            common = common & eligible if common else eligible
        if not common or len(common) < 5:
            return []

        # Volume filter
        if self.min_avg_volume > 0:
            liquid = set()
            for sym in common:
                if sym in price_data and not price_data[sym].empty:
                    df = price_data[sym]
                    if as_of_date is not None:
                        df = df.loc[df.index <= as_of_date]
                    if "volume" in df.columns and len(df) >= 20:
                        if df["volume"].iloc[-20:].mean() >= self.min_avg_volume:
                            liquid.add(sym)
            common = common & liquid if liquid else common

        # Market cap filter
        if fundamental_data and self.min_market_cap > 0:
            big_enough = {
                sym for sym in common
                if fundamental_data.get(sym, {}).get("Market Cap", 0) >= self.min_market_cap
                or fundamental_data.get(sym, {}).get("Market Cap", 0) == 0
            }
            common = common & big_enough

        if len(common) < 5:
            return []

        composites = []
        for sym in common:
            composite_z = 0.0
            factor_scores = {}
            total_w = 0.0
            for fc in self.factors:
                fname = fc.factor.name
                if fname in all_scores and sym in all_scores[fname]:
                    factor_scores[fname] = all_scores[fname][sym]
                    if fc.weight > 0:
                        composite_z += fc.weight * all_scores[fname][sym].z_score
                        total_w += fc.weight
            if total_w > 0:
                composite_z /= total_w
            composites.append(CompositeScore(
                symbol=sym, composite_z=round(composite_z, 4), factor_scores=factor_scores,
            ))

        composites.sort(key=lambda c: c.composite_z, reverse=True)
        for i, c in enumerate(composites):
            c.rank = i + 1
        return composites

    def select_portfolio(self, price_data, fundamental_data=None, sector_map=None, as_of_date=None):
        ranked = self.rank_universe(price_data, fundamental_data, as_of_date)
        if not ranked:
            return []

        selected = []
        sector_counts = {}
        max_per = max(1, int(self.top_n * self.max_sector_pct))

        for c in ranked:
            if len(selected) >= self.top_n:
                break
            sector = "Unknown"
            if sector_map:
                sector = sector_map.get(c.symbol, "Unknown")
            elif fundamental_data and c.symbol in fundamental_data:
                sector = fundamental_data[c.symbol].get("sector", "Unknown")
            count = sector_counts.get(sector, 0)
            if sector != "Unknown" and count >= max_per:
                continue
            c.selected = True
            selected.append(c)
            sector_counts[sector] = count + 1
        return selected

    def compute_weights(self, selected):
        if not selected:
            return {}
        if self.weighting == "score_weighted":
            scores = {s.symbol: s.composite_z for s in selected}
            min_s = min(scores.values())
            shifted = {sym: z - min_s + 0.01 for sym, z in scores.items()}
            total = sum(shifted.values())
            return {sym: round(w / total, 4) for sym, w in shifted.items()}
        n = len(selected)
        w = round(1.0 / n, 4)
        return {s.symbol: w for s in selected}

    def generate_portfolio(self, price_data, fundamental_data=None, sector_map=None, capital=100_000, as_of_date=None):
        selected = self.select_portfolio(price_data, fundamental_data, sector_map, as_of_date)
        weights = self.compute_weights(selected)
        holdings = []
        for cs in selected:
            sym, w = cs.symbol, weights.get(cs.symbol, 0)
            price = 0.0
            if sym in price_data and not price_data[sym].empty:
                df = price_data[sym]
                if as_of_date:
                    df = df.loc[df.index <= as_of_date]
                if not df.empty:
                    price = float(df["close"].iloc[-1])
            shares = int(capital * w / price) if price > 0 else 0
            holdings.append(PortfolioHolding(
                symbol=sym, weight=w, shares=shares, entry_price=price,
                entry_date=str(as_of_date or pd.Timestamp.now())[:10],
                current_price=price, factor_rank=cs.rank,
            ))
        return holdings
