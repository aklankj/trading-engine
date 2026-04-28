"""
factors/quality.py — Cross-sectional quality factors from Screener.in data.
"""
from __future__ import annotations
import numpy as np
from factors.base import BaseFactor


class QualityFactor(BaseFactor):
    """Composite: ROE(25%) + ROCE(25%) + low D/E(20%) + profit growth(15%) + margin(15%)"""
    name = "Quality"
    description = "Composite quality from fundamentals"
    lookback_days = 0
    higher_is_better = True

    def __init__(self, weights=None):
        self.weights = weights or {"roe": 0.25, "roce": 0.25, "low_debt": 0.20, "profit_growth": 0.15, "margin": 0.15}

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        if not fundamental_data:
            return {}
        roe_raw, roce_raw, de_raw, pg_raw, margin_raw = {}, {}, {}, {}, {}
        for sym, fund in fundamental_data.items():
            roe = fund.get("ROE", 0)
            roce = fund.get("ROCE", 0)
            de = fund.get("Debt to equity", fund.get("Debt / Equity", 999))
            pg = fund.get("profit_growth_5 Years", fund.get("profit_growth_5Years", 0))
            opm = fund.get("OPM", roce * 0.8)
            if roe == 0 and roce == 0:
                continue
            roe_raw[sym] = float(roe)
            roce_raw[sym] = float(roce)
            de_raw[sym] = float(de) if de is not None else 999.0
            pg_raw[sym] = float(pg)
            margin_raw[sym] = float(opm)

        common = set(roe_raw) & set(roce_raw) & set(de_raw) & set(pg_raw) & set(margin_raw)
        if len(common) < 10:
            return {}

        def z_dict(d, syms, invert=False):
            vals = np.array([d[s] for s in syms])
            p1, p99 = np.percentile(vals, [1, 99])
            c = np.clip(vals, p1, p99)
            m, s = np.mean(c), np.std(c)
            z = (c - m) / s if s > 1e-10 else np.zeros_like(c)
            if invert:
                z = -z
            return {sym: float(z[i]) for i, sym in enumerate(syms)}

        sym_list = sorted(common)
        roe_z = z_dict(roe_raw, sym_list)
        roce_z = z_dict(roce_raw, sym_list)
        de_z = z_dict(de_raw, sym_list, invert=True)
        pg_z = z_dict(pg_raw, sym_list)
        margin_z = z_dict(margin_raw, sym_list)

        w = self.weights
        return {
            sym: w["roe"] * roe_z[sym] + w["roce"] * roce_z[sym] + w["low_debt"] * de_z[sym]
                 + w["profit_growth"] * pg_z[sym] + w["margin"] * margin_z[sym]
            for sym in sym_list
        }


class LowLeverageFactor(BaseFactor):
    """Inverse debt/equity. Lower debt = higher score."""
    name = "LowLeverage"
    description = "Inverse debt/equity"
    lookback_days = 0
    higher_is_better = False

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        if not fundamental_data:
            return {}
        scores = {}
        for sym, fund in fundamental_data.items():
            de = fund.get("Debt to equity", fund.get("Debt / Equity", None))
            if de is not None and de >= 0:
                scores[sym] = float(de)
        return scores
