"""
Unified transaction cost model for the trading engine.

This module provides a single source of truth for transaction cost calculations
to avoid duplication across the codebase.
"""



def transaction_cost(price: float, quantity: int, side: str) -> float:
    """
    Calculate transaction cost for a trade.
    
    Uses the same 0.1% per side model as BaseStrategy.transaction_cost().
    
    Parameters
    ----------
    price : float
        Execution price of the trade
    quantity : int
        Number of shares traded
    side : str
        "entry" or "exit" to indicate trade type
    
    Returns
    -------
    float
        Transaction cost in rupees
    """
    # Use the same 0.1% model as BaseStrategy.transaction_cost()
    return price * quantity * 0.001


def round_trip_cost(entry_price: float, exit_price: float, quantity: int) -> float:
    """
    Calculate total transaction cost for a round-trip trade.
    
    Parameters
    ----------
    entry_price : float
        Entry price of the trade
    exit_price : float
        Exit price of the trade
    quantity : int
        Number of shares traded
        
    Returns
    -------
    float
        Total transaction cost for the round-trip in rupees
    """
    entry_cost = transaction_cost(entry_price, quantity, "entry")
    exit_cost = transaction_cost(exit_price, quantity, "exit")
    return entry_cost + exit_cost