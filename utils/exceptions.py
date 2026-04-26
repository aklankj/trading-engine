"""
utils/exceptions.py

Structured exception hierarchy for the trading engine.
All exceptions carry optional structured details for logging/debugging.

Hierarchy:
    TradingEngineError (base)
     ├── AuthenticationError   — Kite API auth failures, token expiry
     ├── DataFetchError        — yfinance/Kite historical data failures
     ├── StateCorruptionError  — JSON file corrupted or unreadable
     └── RiskLimitError        — Position size, sector cap, daily loss limit violations
"""

from __future__ import annotations

from typing import Any


class TradingEngineError(Exception):
    """Base exception for all trading engine errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)


class AuthenticationError(TradingEngineError):
    """Kite Connect authentication failures (token expired, login failed)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details, recoverable=True)


class DataFetchError(TradingEngineError):
    """Failed to fetch market data (yfinance, Kite historical, etc.)."""

    def __init__(
        self,
        message: str = "Data fetch failed",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details, recoverable=False)


class StateCorruptionError(TradingEngineError):
    """Persistent JSON state is corrupted or unreadable."""

    def __init__(
        self,
        message: str = "State file corrupted",
        details: dict[str, Any] | None = None,
        repaired: bool = False,
    ):
        self.repaired = repaired
        super().__init__(message, details=details, recoverable=True)


class RiskLimitError(TradingEngineError):
    """Portfolio risk limits breached (position size, sector cap, daily loss)."""

    def __init__(
        self,
        message: str = "Risk limit breached",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details, recoverable=True)