"""
factors/value.py — Cross-sectional value factors from Screener.in data.
"""
from __future__ import annotations
import numpy as np
from factors.base import BaseFactor


class ValueFactor(BaseFactor):
    """Composite: earnings yield (60%) + book-to-price (40%)."""
    name = "Value"
    description = "Earnings yield + book-to-price"
    lookback_days = 0
    higher_is_better = True

    def __init__(self, ey_weight=0.6, bp_weight=0.4):
        self.ey_weight = ey_weight
        self.bp_weight = bp_weight

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        if not fundamental_data:
            return {}
        ey_raw, bp_raw = {}, {}
        for sym, fund in fundamental_data.items():
            pe = fund.get("Stock P/E", 0)
            bv = fund.get("Book Value", 0)
            price = fund.get("Current Price", 0)
            if pe > 0:
                ey_raw[sym] = 1.0 / pe
            if bv > 0 and price > 0:
                bp_raw[sym] = bv / price
        common = set(ey_raw) & set(bp_raw)
        if len(common) < 10:
            return {}

        def z_dict(d, syms):
            vals = np.array([d[s] for s in syms])
            p1, p99 = np.percentile(vals, [1, 99])
            c = np.clip(vals, p1, p99)
            m, s = np.mean(c), np.std(c)
            z = (c - m) / s if s > 1e-10 else np.zeros_like(c)
            return {sym: float(z[i]) for i, sym in enumerate(syms)}

        sym_list = sorted(common)
        ey_z = z_dict(ey_raw, sym_list)
        bp_z = z_dict(bp_raw, sym_list)
        return {sym: self.ey_weight * ey_z[sym] + self.bp_weight * bp_z[sym] for sym in sym_list}
