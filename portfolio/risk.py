"""
portfolio/risk.py

Risk and constraint checks for the portfolio simulator.
Pure functions — no state, no side effects.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import cfg
from utils.logger import log


def compute_sector_exposure(
    open_positions: dict,
    sector_map: dict[str, str],
    data: dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
) -> dict[str, float]:
    """
    Compute current deployed capital per sector.

    Parameters
    ----------
    open_positions : dict[str, SimPosition-like]
        Currently open positions.
    sector_map : dict[str, str]
        Mapping of symbol -> sector name.
    data : dict[str, pd.DataFrame]
        OHLCV data for all symbols.
    current_date : pd.Timestamp
        The current trading date.

    Returns
    -------
    dict[str, float]
        Sector -> total deployed value in rupees.
    """
    exposure: dict[str, float] = {}
    for pos in open_positions.values():
        if current_date in data[pos.symbol].index:
            sec = sector_map.get(pos.symbol, "Unknown")
            price_today = float(data[pos.symbol].loc[current_date, "close"])
            exposure[sec] = exposure.get(sec, 0) + pos.quantity * price_today
    return exposure


def filter_candidates_by_sector(
    candidates: list[tuple],
    sector_map: dict[str, str],
    sector_exposure: dict[str, float],
    cash: float,
    deployed: float,
    position_size_pct: float,
    max_sector_pct: float,
) -> list[tuple]:
    """
    Filter a ranked list of entry candidates so that no sector exceeds
    the maximum allowed exposure.

    Each candidate tuple is (score, sym, strat_name, direction, confidence,
    reason, sl, tgt, entry_price, atr).

    Returns filtered list. Rejected candidates are logged via log.debug.
    """
    filtered: list[tuple] = []
    for cand in candidates:
        sym = cand[1]
        entry_price = cand[8]
        sec = sector_map.get(sym, "Unknown")
        current_sector_val = sector_exposure.get(sec, 0)

        ce = cash + deployed
        new_val = ce * position_size_pct if ce > 0 else 0
        new_sector_pct = ((current_sector_val + new_val) / ce * 100) if ce > 0 else 0

        if new_sector_pct <= max_sector_pct * 100:
            filtered.append(cand)
        else:
            log.debug(
                f"  SKIPPED {sym}: {sec} exposure {new_sector_pct:.0f}% "
                f"> {max_sector_pct * 100:.0f}% cap"
            )
    return filtered


def check_cash_constraint(
    cash: float,
    quantity: int,
    exec_price: float,
    transaction_cost: float,
) -> bool:
    """
    Returns True if the total cost fits within available cash.
    """
    total = quantity * exec_price + transaction_cost
    return total <= cash