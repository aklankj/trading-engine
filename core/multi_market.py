"""
Multi-market data adapter.

Unified interface for fetching OHLCV data across:
  - Indian equities (via Kite Connect)
  - US equities (via yfinance — AAPL, NVDA, TSLA, etc.)
  - Commodities (Gold, Crude, Silver, Copper, Natural Gas)
  - Crypto (BTC, ETH, SOL)
  - Forex (USD/INR, EUR/USD, GBP/USD)
  - Global indices (S&P 500, NASDAQ, DAX, FTSE, Nikkei)

All data is normalized to the same DataFrame format:
  columns: [open, high, low, close, volume]
  index: DatetimeIndex

yfinance is free, no API key needed, covers all global markets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Literal
from dataclasses import dataclass

from utils.logger import log

MarketType = Literal["india", "us", "commodity", "crypto", "forex", "global_index"]


@dataclass
class Instrument:
    """Universal instrument definition."""
    symbol: str            # Display symbol (e.g., "GOLD", "AAPL")
    yf_ticker: str         # yfinance ticker (e.g., "GC=F", "AAPL")
    market: MarketType
    sector: str
    currency: str
    description: str
    kite_token: int = 0    # Only for Indian instruments


# ══════════════════════════════════════════════════════════════
# GLOBAL INSTRUMENT UNIVERSE
# ══════════════════════════════════════════════════════════════

GLOBAL_UNIVERSE: dict[str, Instrument] = {
    # ── US Equities ───────────────────────────────────────────
    "AAPL": Instrument("AAPL", "AAPL", "us", "Tech", "USD", "Apple Inc"),
    "MSFT": Instrument("MSFT", "MSFT", "us", "Tech", "USD", "Microsoft"),
    "NVDA": Instrument("NVDA", "NVDA", "us", "Tech", "USD", "NVIDIA"),
    "GOOGL": Instrument("GOOGL", "GOOGL", "us", "Tech", "USD", "Alphabet"),
    "AMZN": Instrument("AMZN", "AMZN", "us", "Tech", "USD", "Amazon"),
    "TSLA": Instrument("TSLA", "TSLA", "us", "Auto", "USD", "Tesla"),
    "META": Instrument("META", "META", "us", "Tech", "USD", "Meta Platforms"),
    "JPM": Instrument("JPM", "JPM", "us", "Finance", "USD", "JPMorgan Chase"),
    "V": Instrument("V", "V", "us", "Finance", "USD", "Visa"),
    "BRK-B": Instrument("BRK-B", "BRK-B", "us", "Finance", "USD", "Berkshire Hathaway"),

    # ── Commodities ───────────────────────────────────────────
    "GOLD": Instrument("GOLD", "GC=F", "commodity", "Precious Metals", "USD", "Gold Futures"),
    "SILVER": Instrument("SILVER", "SI=F", "commodity", "Precious Metals", "USD", "Silver Futures"),
    "CRUDE": Instrument("CRUDE", "CL=F", "commodity", "Energy", "USD", "WTI Crude Oil"),
    "BRENT": Instrument("BRENT", "BZ=F", "commodity", "Energy", "USD", "Brent Crude"),
    "NATGAS": Instrument("NATGAS", "NG=F", "commodity", "Energy", "USD", "Natural Gas"),
    "COPPER": Instrument("COPPER", "HG=F", "commodity", "Industrial Metals", "USD", "Copper"),
    "WHEAT": Instrument("WHEAT", "ZW=F", "commodity", "Agriculture", "USD", "Wheat"),
    "CORN": Instrument("CORN", "ZC=F", "commodity", "Agriculture", "USD", "Corn"),

    # ── Crypto ────────────────────────────────────────────────
    "BTC": Instrument("BTC", "BTC-USD", "crypto", "Crypto", "USD", "Bitcoin"),
    "ETH": Instrument("ETH", "ETH-USD", "crypto", "Crypto", "USD", "Ethereum"),
    "SOL": Instrument("SOL", "SOL-USD", "crypto", "Crypto", "USD", "Solana"),

    # ── Forex ─────────────────────────────────────────────────
    "USDINR": Instrument("USDINR", "INR=X", "forex", "Currency", "INR", "USD/INR"),
    "EURUSD": Instrument("EURUSD", "EURUSD=X", "forex", "Currency", "USD", "EUR/USD"),
    "GBPUSD": Instrument("GBPUSD", "GBPUSD=X", "forex", "Currency", "USD", "GBP/USD"),
    "USDJPY": Instrument("USDJPY", "JPY=X", "forex", "Currency", "JPY", "USD/JPY"),

    # ── Global Indices ────────────────────────────────────────
    "SP500": Instrument("SP500", "^GSPC", "global_index", "US Index", "USD", "S&P 500"),
    "NASDAQ": Instrument("NASDAQ", "^IXIC", "global_index", "US Index", "USD", "NASDAQ Composite"),
    "DJI": Instrument("DJI", "^DJI", "global_index", "US Index", "USD", "Dow Jones"),
    "FTSE": Instrument("FTSE", "^FTSE", "global_index", "UK Index", "GBP", "FTSE 100"),
    "DAX": Instrument("DAX", "^GDAXI", "global_index", "EU Index", "EUR", "DAX"),
    "NIKKEI": Instrument("NIKKEI", "^N225", "global_index", "Japan Index", "JPY", "Nikkei 225"),
    "HSI": Instrument("HSI", "^HSI", "global_index", "HK Index", "HKD", "Hang Seng"),
    "VIX": Instrument("VIX", "^VIX", "global_index", "Volatility", "USD", "CBOE VIX"),

    # ── India VIX (via yfinance) ──────────────────────────────
    "INDIAVIX": Instrument("INDIAVIX", "^INDIAVIX", "global_index", "Volatility", "INR", "India VIX"),
}


def fetch_global(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for any global instrument.

    Args:
        symbol: Key from GLOBAL_UNIVERSE (e.g., "GOLD", "AAPL", "BTC")
        period: yfinance period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: candle interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

    Returns:
        DataFrame with [open, high, low, close, volume], DatetimeIndex
    """
    import yfinance as yf

    inst = GLOBAL_UNIVERSE.get(symbol)
    if not inst:
        log.warning(f"Unknown symbol: {symbol}")
        return pd.DataFrame()

    try:
        ticker = yf.Ticker(inst.yf_ticker)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            log.warning(f"No data for {symbol} ({inst.yf_ticker})")
            return pd.DataFrame()

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols]

        log.debug(f"Fetched {len(df)} bars for {symbol} ({inst.market})")
        return df

    except Exception as e:
        log.warning(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_multiple(
    symbols: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple instruments efficiently."""
    import yfinance as yf

    # Map to yf tickers
    tickers = {}
    for sym in symbols:
        inst = GLOBAL_UNIVERSE.get(sym)
        if inst:
            tickers[sym] = inst.yf_ticker

    if not tickers:
        return {}

    try:
        # Batch download
        yf_symbols = list(tickers.values())
        data = yf.download(yf_symbols, period=period, interval=interval, group_by="ticker")

        result = {}
        for sym, yf_ticker in tickers.items():
            try:
                if len(yf_symbols) == 1:
                    df = data.copy()
                else:
                    df = data[yf_ticker].copy()
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[keep_cols].dropna()
                if not df.empty:
                    result[sym] = df
            except Exception:
                pass

        log.info(f"Batch fetched {len(result)}/{len(symbols)} instruments")
        return result

    except Exception as e:
        log.warning(f"Batch fetch failed: {e}")
        return {}


def get_macro_dashboard() -> dict:
    """
    Fetch key macro indicators for regime context.
    Returns dict with latest values for major global instruments.
    """
    macro_symbols = [
        "SP500", "VIX", "INDIAVIX", "GOLD", "CRUDE",
        "USDINR", "BTC", "DJI",
    ]

    data = fetch_multiple(macro_symbols, period="5d", interval="1d")

    dashboard = {}
    for sym, df in data.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_pct = (latest["close"] / prev["close"] - 1) if prev["close"] > 0 else 0

        inst = GLOBAL_UNIVERSE[sym]
        dashboard[sym] = {
            "name": inst.description,
            "price": round(latest["close"], 2),
            "change_pct": round(change_pct * 100, 2),
            "currency": inst.currency,
        }

    return dashboard


def get_correlation_matrix(
    symbols: list[str],
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compute rolling correlation matrix between instruments.
    Useful for diversification analysis.
    """
    data = fetch_multiple(symbols, period=period)
    if not data:
        return pd.DataFrame()

    # Build returns DataFrame
    returns = pd.DataFrame()
    for sym, df in data.items():
        if "close" in df.columns:
            returns[sym] = df["close"].pct_change()

    return returns.corr()


def get_instruments_by_market(market: MarketType) -> list[Instrument]:
    """Get all instruments for a given market type."""
    return [i for i in GLOBAL_UNIVERSE.values() if i.market == market]


def search_instrument(query: str) -> list[Instrument]:
    """Search instruments by name or symbol."""
    q = query.lower()
    return [
        i for i in GLOBAL_UNIVERSE.values()
        if q in i.symbol.lower() or q in i.description.lower()
    ]
