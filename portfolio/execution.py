"""
portfolio/execution.py

Entry/exit fill logic for the portfolio simulator.
Pure functions — no state, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from strategies.base import BaseStrategy
from utils.costs import transaction_cost


# ──────────────────────────────────────────
# Fill helpers
# ──────────────────────────────────────────


def fill_entry(
    close_price: float,
    quantity: int,
    slippage_pct: float,
    direction: str,
) -> tuple[float, float, float]:
    """
    Compute the executed entry price and transaction cost after slippage.

    Parameters
    ----------
    close_price : float
        Market close price at entry time.
    quantity : int
        Number of shares to buy/sell.
    slippage_pct : float
        Slippage fraction (e.g. 0.001 for 0.1%).
    direction : str
        "BUY" or "SELL".

    Returns
    -------
    exec_price : float
        Actual fill price after slippage.
        BUY: close_price * (1 + slippage_pct)
        SELL: close_price * (1 - slippage_pct)
    transaction_cost : float
        Brokerage + taxes on entry.
    slippage_cost : float
        Absolute notional cost of slippage.
    """
    if direction == "BUY":
        exec_price = close_price * (1 + slippage_pct)
    else:
        exec_price = close_price * (1 - slippage_pct)

    txn_cost = transaction_cost(exec_price, quantity, "entry")
    slippage_cost = abs(exec_price - close_price) * quantity
    return exec_price, txn_cost, slippage_cost


def fill_exit(
    close_price: float,
    quantity: int,
    slippage_pct: float,
    direction: str,
) -> tuple[float, float, float]:
    """
    Compute the executed exit price, costs, and P&L after slippage.

    Returns
    -------
    exec_price : float
        Actual fill price after slippage.
    transaction_cost : float
        Brokerage + taxes on exit.
    slippage_cost : float
        Absolute notional cost of slippage.
    pnl : float
        Gross P&L on the trade (before exit transaction cost).
    pnl_pct : float
        Percentage return.
    """
    if direction == "BUY":
        exec_price = close_price * (1 - slippage_pct)
    else:
        exec_price = close_price * (1 + slippage_pct)

    txn_cost = transaction_cost(exec_price, quantity, "exit")
    slippage_cost = abs(close_price - exec_price) * quantity

    # P&L is computed against the entry price stored on the position
    # (caller must pass the correct entry_price)
    return exec_price, txn_cost, slippage_cost


def compute_trade_pnl(
    exec_price: float,
    entry_price: float,
    quantity: int,
    direction: str,
    transaction_cost: float,
) -> tuple[float, float]:
    """
    Compute net P&L and percentage return for a closed trade.

    Parameters
    ----------
    exec_price : float
        The fill price at exit.
    entry_price : float
        The fill price at entry.
    quantity : int
        Number of shares.
    direction : str
        "BUY" or "SELL".
    transaction_cost : float
        Exit transaction cost (deducted from P&L).

    Returns
    -------
    pnl : float
        Net P&L in rupees.
    pnl_pct : float
        Percentage return (relative to entry).
    """
    if direction == "BUY":
        gross_pnl = (exec_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exec_price) * quantity

    pnl = gross_pnl - transaction_cost
    pnl_pct = (exec_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    if direction == "SELL":
        pnl_pct = (entry_price - exec_price) / entry_price * 100 if entry_price > 0 else 0.0

    return round(pnl, 2), round(pnl_pct, 2)