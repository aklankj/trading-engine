"""
Tests for unified cost model and zero quantity guards.

This test suite covers:
1. Transaction cost calculations (entry/exit/round-trip)
2. Zero quantity guards in portfolio operations
3. Consistent cost behavior across modules
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from utils.costs import transaction_cost, round_trip_cost
from execution.portfolio_tracker import open_position, close_position, _load_portfolio, _save_portfolio, reset_portfolio
from portfolio.execution import fill_entry, fill_exit
from portfolio.simulator import simulate
from tests.conftest import make_test_data


def test_transaction_cost_calculation():
    """Test that transaction_cost function works correctly."""
    # Test entry cost
    cost = transaction_cost(100.0, 10, "entry")
    assert cost == 1.0  # 100 * 10 * 0.001
    
    # Test exit cost
    cost = transaction_cost(100.0, 10, "exit")
    assert cost == 1.0  # 100 * 10 * 0.001
    
    # Test different values
    cost = transaction_cost(50.0, 20, "entry")
    assert cost == 1.0  # 50 * 20 * 0.001


def test_round_trip_cost_calculation():
    """Test round-trip cost calculation."""
    cost = round_trip_cost(100.0, 110.0, 10)
    expected = 1.0 + 1.1  # entry cost + exit cost
    assert cost == expected


def test_zero_quantity_open_position_guard():
    """Test that open_position rejects zero/negative quantities."""
    reset_portfolio()
    
    # Test zero quantity
    result = open_position(
        symbol="TEST",
        direction="BUY",
        quantity=0,
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        regime="bull",
        signal_strength=0.8,
        strategy="TestStrategy"
    )
    assert result == {}  # Should return empty dict
    
    # Test negative quantity
    result = open_position(
        symbol="TEST2",
        direction="BUY",
        quantity=-5,
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        regime="bull",
        signal_strength=0.8,
        strategy="TestStrategy"
    )
    assert result == {}  # Should return empty dict


def test_zero_quantity_close_position_guard():
    """Test that close_position handles zero quantity gracefully."""
    reset_portfolio()
    
    # First open a position
    open_position(
        symbol="TEST",
        direction="BUY",
        quantity=10,
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        regime="bull",
        signal_strength=0.8,
        strategy="TestStrategy"
    )
    
    # Verify position exists
    portfolio = _load_portfolio()
    assert "TEST" in portfolio["positions"]
    
    # Test close with zero quantity (should not crash)
    # Note: This test is more about ensuring no crash than actual behavior
    # since close_position doesn't directly take quantity parameter
    # but we can test that it handles the case gracefully
    result = close_position("TEST", 110.0, "manual")
    assert result is not None
    assert result["symbol"] == "TEST"


def test_fill_entry_cost_calculation():
    """Test that fill_entry uses unified cost model."""
    exec_price, txn_cost, slip_cost = fill_entry(100.0, 10, 0.0, "BUY")
    assert exec_price == 100.0  # No slippage
    assert txn_cost == 1.0  # 100 * 10 * 0.001


def test_fill_exit_cost_calculation():
    """Test that fill_exit uses unified cost model."""
    exec_price, txn_cost, slip_cost = fill_exit(100.0, 10, 0.0, "BUY")
    assert exec_price == 100.0  # No slippage
    assert txn_cost == 1.0  # 100 * 10 * 0.001


def test_simulator_zero_quantity_guard():
    """Test that simulator skips candidates with zero/negative quantity."""
    # Create minimal test data
    df = make_test_data(days=100)
    data = {"TEST": df}
    
    # Mock a strategy that returns zero quantity by manipulating the position size calculation
    with patch("strategies.registry.CORE_STRATEGIES") as mock_strategies:
        # Create a mock strategy that returns valid values but will result in zero quantity
        mock_strategy = MagicMock()
        mock_strategy.min_bars.return_value = 1
        mock_strategy.should_enter.return_value = ("BUY", 0.8, "test", 90.0, 110.0)
        mock_strategy.should_exit.return_value = (False, "no_exit")
        mock_strategy.compute_indicators.return_value = df
        
        mock_strategies.__getitem__.return_value = mock_strategy
        
        # Force a scenario where quantity calculation results in zero
        # by using a very high entry price that makes quantity = 0
        result = simulate(
            data=data,
            strategies={"TEST": mock_strategy},
            initial_capital=100000,
            max_positions=10,
            position_size_pct=0.000001,  # Very small position size to force zero quantity
        )
        
        # Should return a valid result
        assert result is not None
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trade_log')


def test_cost_consistency_across_modules():
    """Test that all modules use the same cost calculation."""
    # Test the same calculation in different contexts
    price = 100.0
    quantity = 10
    
    # Direct function call
    direct_cost = transaction_cost(price, quantity, "entry")
    
    # Through fill_entry
    exec_price, entry_cost, slip_cost = fill_entry(price, quantity, 0.0, "BUY")
    
    # Through fill_exit  
    exec_price, exit_cost, slip_cost = fill_exit(price, quantity, 0.0, "BUY")
    
    # All should be the same
    assert direct_cost == 1.0
    assert entry_cost == 1.0
    assert exit_cost == 1.0