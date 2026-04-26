"""
Data fetching from Kite Connect.

Provides clean pandas DataFrames for:
- Historical OHLCV candles (any interval)
- Live quotes (LTP, OHLC, market depth)
- Portfolio holdings and positions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Literal

from core.auth import get_kite
from utils.logger import log

IntervalType = Literal[
    "minute", "3minute", "5minute", "10minute",
    "15minute", "30minute", "60minute", "day"
]

# Max days per request by interval (Kite API limits)
_INTERVAL_LIMITS = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}


def fetch_candles(
    instrument_token: int,
    interval: IntervalType = "day",
    days: int = 120,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles as a DataFrame.

    Automatically handles Kite's per-request day limits by chunking.

    Returns DataFrame with columns: [date, open, high, low, close, volume]
    Index: DatetimeIndex (IST)
    """
    kite = get_kite()
    to_dt = to_date or datetime.now()
    from_dt = from_date or (to_dt - timedelta(days=days))

    max_days = _INTERVAL_LIMITS.get(interval, 60)
    all_records = []

    # Chunk requests to stay within limits
    chunk_start = from_dt
    while chunk_start < to_dt:
        chunk_end = min(chunk_start + timedelta(days=max_days), to_dt)
        try:
            records = kite.historical_data(
                instrument_token=instrument_token,
                from_date=chunk_start,
                to_date=chunk_end,
                interval=interval,
            )
            all_records.extend(records)
        except Exception as e:
            log.warning(f"Candle fetch failed for chunk {chunk_start}-{chunk_end}: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not all_records:
        log.warning(f"No candle data for token {instrument_token}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    log.debug(f"Fetched {len(df)} candles for token {instrument_token} ({interval})")
    return df


def fetch_quotes(symbols: list[str]) -> dict:
    """
    Fetch live quotes for a list of symbols.

    Args:
        symbols: List like ["NSE:RELIANCE", "NSE:INFY"]

    Returns:
        Dict keyed by symbol with quote data (ltp, ohlc, volume, etc.)
    """
    kite = get_kite()
    try:
        return kite.quote(symbols)
    except Exception as e:
        log.error(f"Quote fetch failed: {e}")
        return {}


def fetch_ltp(symbols: list[str]) -> dict[str, float]:
    """
    Fetch last traded prices.

    Returns:
        Dict of symbol -> price, e.g. {"NSE:RELIANCE": 2450.50}
    """
    kite = get_kite()
    try:
        data = kite.ltp(symbols)
        return {sym: info["last_price"] for sym, info in data.items()}
    except Exception as e:
        log.error(f"LTP fetch failed: {e}")
        return {}


def fetch_holdings() -> pd.DataFrame:
    """Fetch current holdings (CNC delivery positions)."""
    kite = get_kite()
    try:
        holdings = kite.holdings()
        if not holdings:
            return pd.DataFrame()
        df = pd.DataFrame(holdings)
        log.info(f"Fetched {len(df)} holdings")
        return df
    except Exception as e:
        log.error(f"Holdings fetch failed: {e}")
        return pd.DataFrame()


def fetch_positions() -> pd.DataFrame:
    """Fetch current day positions (MIS/NRML)."""
    kite = get_kite()
    try:
        positions = kite.positions()
        day_pos = positions.get("day", [])
        net_pos = positions.get("net", [])
        combined = day_pos + net_pos
        if not combined:
            return pd.DataFrame()
        df = pd.DataFrame(combined).drop_duplicates(
            subset=["tradingsymbol", "exchange", "product"]
        )
        return df
    except Exception as e:
        log.error(f"Positions fetch failed: {e}")
        return pd.DataFrame()


def fetch_margins() -> dict:
    """Fetch available margins."""
    kite = get_kite()
    try:
        return kite.margins()
    except Exception as e:
        log.error(f"Margins fetch failed: {e}")
        return {}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a candle DataFrame.
    Expects columns: open, high, low, close, volume
    """
    if df.empty or len(df) < 50:
        return df

    c = df["close"]

    # Moving averages
    df["sma_20"] = c.rolling(20).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (14-period)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    df["bb_mid"] = df["sma_20"]
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR (14-period)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - c.shift()).abs()
    low_close = (df["low"] - c.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # Donchian Channel (20-period)
    df["dc_high"] = df["high"].rolling(20).max()
    df["dc_low"] = df["low"].rolling(20).min()
    df["dc_mid"] = (df["dc_high"] + df["dc_low"]) / 2

    # Returns
    df["returns"] = c.pct_change()
    df["log_returns"] = np.log(c / c.shift())
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)

    return df
