"""Tests for portfolio/risk.py — risk constraint functions."""

from __future__ import annotations

import pytest

from portfolio.risk import (
    compute_sector_exposure,
    filter_candidates_by_sector,
    check_cash_constraint,
)
from utils.exceptions import RiskLimitError


def test_filter_candidates_by_sector_removes_over_cap():
    """If a candidate would exceed the sector cap, it should be removed."""
    # Test 1: Distinct sectors with reasonable cap — both pass
    candidates = [
        (1.0, "IT_STOCK", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
        (0.9, "BANK_STOCK", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
    ]
    sector_map = {"IT_STOCK": "IT", "BANK_STOCK": "Banking"}
    filtered = filter_candidates_by_sector(
        candidates, sector_map, {"IT": 0, "Banking": 0},
        cash=100_000, deployed=0, position_size_pct=0.05,
        max_sector_pct=0.30,
    )
    assert len(filtered) == 2

    # Test 2: Same sector, position size > sector cap — blocked
    single_sector = [
        (1.0, "A", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
        (0.9, "B", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
    ]
    single_map = {"A": "Tech", "B": "Tech"}
    # 6% position, 5% cap — each individually > cap, all blocked
    filtered_blocked = filter_candidates_by_sector(
        single_sector, single_map, {"Tech": 0},
        cash=100_000, deployed=0, position_size_pct=0.06,
        max_sector_pct=0.05,
    )
    assert len(filtered_blocked) == 0

    # Test 3: Same sector, position size < cap — each passes individually
    filtered_pass = filter_candidates_by_sector(
        single_sector, single_map, {"Tech": 0},
        cash=100_000, deployed=0, position_size_pct=0.03,
        max_sector_pct=0.05,
    )
    assert len(filtered_pass) == 2


def test_empty_sector_map_no_filter():
    """With empty sector_map, all candidates should pass unchanged."""
    candidates = [
        (1.0, "A", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
        (0.9, "B", "T", "BUY", 0.5, "test", 90.0, 110.0, 100.0, 2.0),
    ]
    filtered = filter_candidates_by_sector(
        candidates, {}, {}, cash=100_000, deployed=0,
        position_size_pct=0.05, max_sector_pct=0.30,
    )
    assert len(filtered) == 2


def test_risk_limit_error_can_be_raised():
    """RiskLimitError should carry structured details."""
    try:
        raise RiskLimitError("Sector cap exceeded", details={"sector": "IT", "exposure": 0.35, "limit": 0.30})
    except RiskLimitError as e:
        assert e.details["sector"] == "IT"
        assert e.details["exposure"] == 0.35
        assert e.recoverable is True


def test_check_cash_constraint():
    """check_cash_constraint should verify if total cost fits in cash."""
    assert check_cash_constraint(cash=100_000, quantity=10, exec_price=100.0, transaction_cost=50.0) is True
    assert check_cash_constraint(cash=1000, quantity=10, exec_price=100.0, transaction_cost=50.0) is False