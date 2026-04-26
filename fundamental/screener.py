"""
Fundamental quality screener.

Scores companies on 9 weighted metrics and detects
buy opportunities when quality stocks drop into value territory.

Data sources:
  - Screener.in (manual CSV export or scrape)
  - Kite Connect (live prices, 52-week high/low)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from config.settings import cfg
from utils.logger import log
from utils import load_json, save_json, fmt_inr


@dataclass
class QualityScore:
    """Scored company with buy/hold/watch signal."""
    name: str
    symbol: str
    exchange: str
    sector: str
    score: float           # 0-100
    signal: str            # "buy-dip", "hold", "watch", "avoid"
    roce: float
    roic: float
    fcf_ratio: float       # FCF / Net Profit
    debt_equity: float
    rev_cagr_5y: float
    opm: float
    promoter_holding: float
    pe_vs_sector: float    # <1 = cheaper than sector median
    interest_coverage: float
    note: str
    current_price: float
    high_52w: float
    drop_from_high_pct: float


# ── Quality criteria with weights ─────────────────────────────

CRITERIA = [
    {"metric": "roce",              "weight": 25, "good": lambda v: v > 15, "great": lambda v: v > 25},
    {"metric": "roic",              "weight": 15, "good": lambda v: v > 12, "great": lambda v: v > 20},
    {"metric": "fcf_ratio",         "weight": 15, "good": lambda v: v > 0.8, "great": lambda v: v > 0.95},
    {"metric": "debt_equity",       "weight": 10, "good": lambda v: v < 0.5, "great": lambda v: v < 0.1},
    {"metric": "rev_cagr_5y",       "weight": 10, "good": lambda v: v > 10, "great": lambda v: v > 18},
    {"metric": "opm",               "weight": 10, "good": lambda v: v > 15, "great": lambda v: v > 25},
    {"metric": "promoter_holding",  "weight": 5,  "good": lambda v: v > 50, "great": lambda v: v > 65},
    {"metric": "pe_vs_sector",      "weight": 5,  "good": lambda v: v < 1.0, "great": lambda v: v < 0.7},
    {"metric": "interest_coverage", "weight": 5,  "good": lambda v: v > 5, "great": lambda v: v > 15},
]


def compute_quality_score(data: dict) -> float:
    """
    Compute weighted quality score (0-100) from fundamental data.

    data should contain keys matching CRITERIA metrics.
    """
    total = 0.0
    for c in CRITERIA:
        metric = c["metric"]
        val = data.get(metric, 0)
        weight = c["weight"]

        if c["great"](val):
            total += weight * 1.0
        elif c["good"](val):
            total += weight * 0.7
        else:
            total += weight * 0.3

    return round(total, 1)


def load_fundamental_watchlist() -> list[dict]:
    """
    Load the fundamental watchlist from JSON file.
    Each entry should have company fundamentals.
    """
    path = cfg.DATA_DIR / "fundamental_watchlist.json"
    if path.exists():
        return load_json(path, default=[])

    # Return default watchlist if no file exists
    return _default_fundamental_watchlist()


def save_fundamental_watchlist(watchlist: list[dict]):
    """Save updated watchlist."""
    path = cfg.DATA_DIR / "fundamental_watchlist.json"
    save_json(path, watchlist)


def _default_fundamental_watchlist() -> list[dict]:
    """Starter watchlist with approximate fundamental data."""
    return [
        {"name": "HDFC Bank", "symbol": "HDFCBANK", "exchange": "NSE", "sector": "Banking",
         "roce": 16.8, "roic": 14.2, "fcf_ratio": 0.92, "debt_equity": 0.12,
         "rev_cagr_5y": 18.5, "opm": 28.3, "promoter_holding": 25.5,
         "pe_vs_sector": 0.95, "interest_coverage": 18.0,
         "note": "Largest private bank. Excellent quality, usually fully priced."},
        {"name": "Asian Paints", "symbol": "ASIANPAINT", "exchange": "NSE", "sector": "Consumer",
         "roce": 28.4, "roic": 22.1, "fcf_ratio": 0.95, "debt_equity": 0.08,
         "rev_cagr_5y": 12.3, "opm": 18.7, "promoter_holding": 52.6,
         "pe_vs_sector": 1.2, "interest_coverage": 45.0,
         "note": "Dominant paint franchise. Reinvestment >50% at ROCE >25%."},
        {"name": "Bajaj Finance", "symbol": "BAJFINANCE", "exchange": "NSE", "sector": "NBFC",
         "roce": 21.5, "roic": 16.8, "fcf_ratio": 0.78, "debt_equity": 0.35,
         "rev_cagr_5y": 24.1, "opm": 32.1, "promoter_holding": 54.7,
         "pe_vs_sector": 0.85, "interest_coverage": 8.0,
         "note": "Growth compounder in consumer lending."},
        {"name": "TCS", "symbol": "TCS", "exchange": "NSE", "sector": "IT",
         "roce": 52.3, "roic": 45.2, "fcf_ratio": 1.05, "debt_equity": 0.02,
         "rev_cagr_5y": 9.8, "opm": 25.1, "promoter_holding": 72.3,
         "pe_vs_sector": 1.1, "interest_coverage": 120.0,
         "note": "Capital-light IT giant. Consistent cash generator."},
        {"name": "Pidilite", "symbol": "PIDILITIND", "exchange": "NSE", "sector": "Chemicals",
         "roce": 26.7, "roic": 20.3, "fcf_ratio": 0.88, "debt_equity": 0.05,
         "rev_cagr_5y": 14.2, "opm": 22.4, "promoter_holding": 69.9,
         "pe_vs_sector": 1.3, "interest_coverage": 55.0,
         "note": "Monopoly in adhesives (Fevicol). Strong reinvestment."},
        {"name": "Titan", "symbol": "TITAN", "exchange": "NSE", "sector": "Consumer",
         "roce": 24.1, "roic": 18.9, "fcf_ratio": 0.72, "debt_equity": 0.15,
         "rev_cagr_5y": 21.8, "opm": 12.1, "promoter_holding": 52.9,
         "pe_vs_sector": 0.9, "interest_coverage": 22.0,
         "note": "Massive runway in jewelry + watches. Tata-backed."},
        {"name": "CDSL", "symbol": "CDSL", "exchange": "NSE", "sector": "Capital Markets",
         "roce": 42.8, "roic": 38.1, "fcf_ratio": 1.12, "debt_equity": 0.01,
         "rev_cagr_5y": 28.4, "opm": 62.5, "promoter_holding": 20.0,
         "pe_vs_sector": 1.5, "interest_coverage": 200.0,
         "note": "Infrastructure monopoly in depository services."},
        {"name": "Dmart", "symbol": "DMART", "exchange": "NSE", "sector": "Retail",
         "roce": 19.3, "roic": 15.5, "fcf_ratio": 0.65, "debt_equity": 0.03,
         "rev_cagr_5y": 22.5, "opm": 8.2, "promoter_holding": 74.6,
         "pe_vs_sector": 1.4, "interest_coverage": 80.0,
         "note": "Low-margin high-volume play. Debt-free. Long runway."},
    ]


# ── Red flag detection ────────────────────────────────────────

RED_FLAGS = [
    {"name": "ROCE declining", "check": lambda d: d.get("roce", 0) < 12,
     "msg": "ROCE below 12% — capital efficiency weakening"},
    {"name": "High debt", "check": lambda d: d.get("debt_equity", 0) > 1.0,
     "msg": "Debt/Equity above 1.0 — overleveraged"},
    {"name": "Low FCF conversion", "check": lambda d: d.get("fcf_ratio", 0) < 0.5,
     "msg": "FCF/Profit below 0.5 — earnings may not be real cash"},
    {"name": "Promoter pledging", "check": lambda d: d.get("promoter_pledge_pct", 0) > 20,
     "msg": "Promoter pledge above 20% — governance risk"},
    {"name": "Negative growth", "check": lambda d: d.get("rev_cagr_5y", 0) < 0,
     "msg": "Revenue declining — shrinking business"},
]


def check_red_flags(data: dict) -> list[str]:
    """Return list of red flag messages for a company."""
    flags = []
    for rf in RED_FLAGS:
        if rf["check"](data):
            flags.append(rf["msg"])
    return flags


def score_watchlist(watchlist: list[dict] = None) -> list[QualityScore]:
    """
    Score all companies in the watchlist and determine signals.
    """
    if watchlist is None:
        watchlist = load_fundamental_watchlist()

    results = []
    for company in watchlist:
        score = compute_quality_score(company)
        red_flags = check_red_flags(company)

        # Determine signal based on score and red flags
        if red_flags:
            signal = "avoid"
            note = f"RED FLAGS: {'; '.join(red_flags)}"
        elif score >= 85:
            signal = "buy-dip"
            note = company.get("note", "High quality — buy on significant dips")
        elif score >= 70:
            signal = "hold"
            note = company.get("note", "Good quality — hold existing positions")
        elif score >= 55:
            signal = "watch"
            note = company.get("note", "Decent quality — monitor for improvement")
        else:
            signal = "avoid"
            note = "Quality score too low"

        results.append(QualityScore(
            name=company["name"],
            symbol=company["symbol"],
            exchange=company.get("exchange", "NSE"),
            sector=company.get("sector", "Unknown"),
            score=score,
            signal=signal,
            roce=company.get("roce", 0),
            roic=company.get("roic", 0),
            fcf_ratio=company.get("fcf_ratio", 0),
            debt_equity=company.get("debt_equity", 0),
            rev_cagr_5y=company.get("rev_cagr_5y", 0),
            opm=company.get("opm", 0),
            promoter_holding=company.get("promoter_holding", 0),
            pe_vs_sector=company.get("pe_vs_sector", 1.0),
            interest_coverage=company.get("interest_coverage", 0),
            note=note,
            current_price=0,  # Filled by buy_trigger check
            high_52w=0,
            drop_from_high_pct=0,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    log.info(f"Scored {len(results)} companies | "
             f"Buy-dip: {sum(1 for r in results if r.signal == 'buy-dip')} | "
             f"Hold: {sum(1 for r in results if r.signal == 'hold')} | "
             f"Watch: {sum(1 for r in results if r.signal == 'watch')}")
    return results
