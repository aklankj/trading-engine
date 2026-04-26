"""
Correlation guard + transaction cost modeling.

Correlation Guard:
  Before opening a new position, checks how correlated it is
  with existing holdings. Prevents hidden concentrated bets
  (e.g., HDFCBANK + ICICIBANK + KOTAKBANK = 3x banking exposure).

Transaction Costs:
  Models realistic all-in costs for Indian and US markets:
  - Brokerage, STT, exchange fees, GST, stamp duty
  - Slippage (market impact based on volume)
  - Spread cost
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from utils.logger import log


# ══════════════════════════════════════════════════════════════
# CORRELATION GUARD
# ══════════════════════════════════════════════════════════════

@dataclass
class CorrelationCheck:
    approved: bool
    avg_correlation: float
    max_correlation: float
    max_corr_with: str
    adjusted_size_mult: float  # 0.0-1.0, multiply position size by this
    reason: str


def check_correlation(
    new_symbol: str,
    new_returns: pd.Series,
    open_positions: dict[str, pd.Series],
    max_avg_corr: float = 0.7,
    max_single_corr: float = 0.85,
    lookback: int = 20,
) -> CorrelationCheck:
    """
    Check if a new position is too correlated with existing holdings.

    Args:
        new_symbol: Symbol being considered
        new_returns: Daily returns series for the new symbol
        open_positions: Dict of symbol -> daily returns for open positions
        max_avg_corr: Max average correlation with all open positions
        max_single_corr: Max correlation with any single position
        lookback: Rolling correlation window (days)

    Returns:
        CorrelationCheck with approval and size adjustment
    """
    if not open_positions:
        return CorrelationCheck(True, 0.0, 0.0, "", 1.0, "No open positions")

    correlations = {}
    for sym, returns in open_positions.items():
        # Align the two series
        aligned = pd.concat([new_returns, returns], axis=1).dropna()
        if len(aligned) < lookback:
            correlations[sym] = 0.0
            continue

        recent = aligned.iloc[-lookback:]
        corr = recent.iloc[:, 0].corr(recent.iloc[:, 1])
        correlations[sym] = abs(corr) if not np.isnan(corr) else 0.0

    if not correlations:
        return CorrelationCheck(True, 0.0, 0.0, "", 1.0, "Insufficient data")

    avg_corr = np.mean(list(correlations.values()))
    max_corr = max(correlations.values())
    max_corr_sym = max(correlations, key=correlations.get)

    # Decision logic
    if max_corr >= max_single_corr:
        return CorrelationCheck(
            False, avg_corr, max_corr, max_corr_sym, 0.0,
            f"Too correlated with {max_corr_sym} ({max_corr:.2f})"
        )

    if avg_corr >= max_avg_corr:
        return CorrelationCheck(
            False, avg_corr, max_corr, max_corr_sym, 0.0,
            f"Average correlation too high ({avg_corr:.2f})"
        )

    # Partial size reduction for moderate correlation
    if avg_corr > 0.5:
        size_mult = 1.0 - (avg_corr - 0.5) * 2  # Linear reduction from 1.0 to 0.0
        size_mult = max(0.3, size_mult)
        return CorrelationCheck(
            True, avg_corr, max_corr, max_corr_sym, size_mult,
            f"Moderate correlation — size reduced to {size_mult:.0%}"
        )

    return CorrelationCheck(
        True, avg_corr, max_corr, max_corr_sym, 1.0,
        f"Low correlation — full size approved"
    )


def compute_portfolio_correlation_matrix(
    returns_dict: dict[str, pd.Series],
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Build a correlation matrix for the entire portfolio.
    Useful for daily monitoring and rebalancing decisions.
    """
    if not returns_dict:
        return pd.DataFrame()

    df = pd.DataFrame(returns_dict)
    recent = df.iloc[-lookback:] if len(df) > lookback else df
    return recent.corr()


# ══════════════════════════════════════════════════════════════
# TRANSACTION COST MODELING
# ══════════════════════════════════════════════════════════════

@dataclass
class TransactionCost:
    """Breakdown of all trading costs for a single trade."""
    brokerage: float
    stt: float                # Securities Transaction Tax
    exchange_fees: float
    gst: float
    stamp_duty: float
    sebi_fees: float
    slippage_est: float       # Estimated slippage
    total: float
    total_pct: float          # Total cost as % of trade value


def estimate_costs_india(
    value: float,
    quantity: int,
    is_buy: bool,
    product: str = "MIS",
    avg_daily_volume: float = 1e6,
) -> TransactionCost:
    """
    Estimate realistic all-in transaction costs for NSE trades.

    Based on Zerodha's fee structure (March 2025):
      - Brokerage: ₹0 for equity delivery, ₹20 or 0.03% for intraday
      - STT: 0.1% on sell (delivery), 0.025% on sell (intraday)
      - Exchange: 0.00345% (NSE)
      - GST: 18% on (brokerage + exchange)
      - Stamp: 0.015% on buy (delivery), 0.003% on buy (intraday)
      - SEBI: ₹10 per crore

    Slippage model: based on order size relative to avg daily volume.
    """
    is_delivery = product == "CNC"

    # Brokerage
    if is_delivery:
        brokerage = 0.0
    else:
        brokerage = min(20.0, value * 0.0003)

    # STT (only on sell side)
    if not is_buy:
        stt = value * (0.001 if is_delivery else 0.00025)
    else:
        stt = value * 0.001 if is_delivery else 0.0

    # Exchange transaction charges
    exchange_fees = value * 0.0000345

    # GST (18% on brokerage + exchange)
    gst = (brokerage + exchange_fees) * 0.18

    # Stamp duty (only on buy)
    if is_buy:
        stamp = value * (0.00015 if is_delivery else 0.00003)
    else:
        stamp = 0.0

    # SEBI turnover fees
    sebi = value * 0.000001  # ₹10 per crore

    # Slippage estimation
    # Model: slippage increases with order size / daily volume
    participation_rate = value / (avg_daily_volume * 0.01) if avg_daily_volume > 0 else 0
    base_slippage_pct = 0.05  # 0.05% base for liquid stocks
    slippage_pct = base_slippage_pct + participation_rate * 0.02
    slippage_pct = min(slippage_pct, 0.5)  # Cap at 0.5%
    slippage = value * slippage_pct / 100

    total = brokerage + stt + exchange_fees + gst + stamp + sebi + slippage
    total_pct = total / value * 100 if value > 0 else 0

    return TransactionCost(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange_fees=round(exchange_fees, 2),
        gst=round(gst, 2),
        stamp_duty=round(stamp, 2),
        sebi_fees=round(sebi, 2),
        slippage_est=round(slippage, 2),
        total=round(total, 2),
        total_pct=round(total_pct, 4),
    )


def estimate_costs_us(
    value: float,
    quantity: int,
    is_buy: bool,
) -> TransactionCost:
    """
    Estimate costs for US equity trades (via international brokers).
    Most brokers charge $0-$1 per trade for equities.
    Main cost is forex conversion (~0.3-0.5% on INR/USD).
    """
    brokerage = 1.0  # Flat $1
    forex_cost = value * 0.004  # ~0.4% forex spread
    sec_fee = value * 0.0000278 if not is_buy else 0  # SEC fee on sells
    slippage = value * 0.0003  # 0.03% for liquid US stocks

    total = brokerage + forex_cost + sec_fee + slippage
    total_pct = total / value * 100 if value > 0 else 0

    return TransactionCost(
        brokerage=round(brokerage, 2),
        stt=round(sec_fee, 2),
        exchange_fees=0,
        gst=0,
        stamp_duty=0,
        sebi_fees=0,
        slippage_est=round(slippage + forex_cost, 2),
        total=round(total, 2),
        total_pct=round(total_pct, 4),
    )


def estimate_roundtrip_cost(
    value: float,
    market: str = "india",
    product: str = "MIS",
    avg_daily_volume: float = 1e6,
) -> float:
    """
    Estimate total roundtrip cost (buy + sell) as a percentage.
    This is the minimum the trade must profit to break even.
    """
    if market == "india":
        buy = estimate_costs_india(value, 1, True, product, avg_daily_volume)
        sell = estimate_costs_india(value, 1, False, product, avg_daily_volume)
    else:
        buy = estimate_costs_us(value, 1, True)
        sell = estimate_costs_us(value, 1, False)

    return buy.total_pct + sell.total_pct
