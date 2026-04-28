"""
factors/universe.py — Stock universe, sector map, and data fetching.
"""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
    "ITC", "SBIN", "LT", "HCLTECH", "BAJFINANCE", "HINDUNILVR",
    "MARUTI", "KOTAKBANK", "ADANIENT", "NTPC", "SUNPHARMA", "TATAMOTORS",
    "ONGC", "WIPRO", "M&M", "ULTRACEMCO", "AXISBANK", "POWERGRID",
    "ASIANPAINT", "TITAN", "NESTLEIND", "JSWSTEEL", "TATASTEEL",
    "BAJAJ-AUTO", "TECHM", "INDUSINDBK", "APOLLOHOSP", "CIPLA",
    "DRREDDY", "EICHERMOT", "BPCL", "COALINDIA", "DIVISLAB",
    "BRITANNIA", "HEROMOTOCO", "HINDALCO", "GRASIM", "TRENT",
    "SHRIRAMFIN", "ADANIPORTS", "PIDILITIND", "BAJAJFINSV", "TATACONSUM",
    "SBILIFE",
]
NIFTY_NEXT50 = [
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
    "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "CONCOR", "DABUR", "DLF", "GAIL",
    "GODREJCP", "HAVELLS", "HAL", "ICICIPRULI", "IDFCFIRSTB",
    "INDHOTEL", "INDUSTOWER", "IOC", "IRCTC", "IRFC",
    "JIOFIN", "JINDALSTEL", "JSWENERGY", "LICI", "LODHA",
    "LUPIN", "MARICO", "MAXHEALTH", "MOTHERSON", "NAUKRI",
    "NHPC", "OBEROIRLTY", "OFSS", "PAYTM", "PFC",
    "PIIND", "PNB", "POLYCAB", "RECLTD", "SBICARD",
    "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM", "TVSMOTOR",
    "UNIONBANK", "UNITDSPR", "VEDL", "ZOMATO", "ZYDUSLIFE",
]
MIDCAP_SELECT = [
    "ASTRAL", "ATUL", "BALKRISIND", "BATAINDIA", "CANFINHOME",
    "CDSL", "CUMMINSIND", "DEEPAKNTR", "DMART", "ESCORTS",
    "FEDERALBNK", "FORTIS", "GMRINFRA", "IPCALAB", "JUBLFOOD",
    "LALPATHLAB", "LICHSGFIN", "LTIM", "LTTS", "MFSL",
    "MPHASIS", "MUTHOOTFIN", "PAGEIND", "PERSISTENT", "PETRONET",
    "PHOENIXLTD", "PVRINOX", "RAMCOCEM", "SAIL", "SYNGENE",
    "TATACHEM", "TATACOMM", "TATAELXSI", "TIINDIA", "TRIDENT",
    "VOLTAS", "YESBANK",
]

SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "LTTS": "IT", "MPHASIS": "IT", "PERSISTENT": "IT",
    "TATAELXSI": "IT", "OFSS": "IT", "NAUKRI": "IT",
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks",
    "KOTAKBANK": "Banks", "AXISBANK": "Banks", "INDUSINDBK": "Banks",
    "BANKBARODA": "Banks", "CANBK": "Banks", "PNB": "Banks",
    "IDFCFIRSTB": "Banks", "FEDERALBNK": "Banks", "YESBANK": "Banks",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "SHRIRAMFIN": "NBFC",
    "CHOLAFIN": "NBFC", "MFSL": "NBFC", "MUTHOOTFIN": "NBFC",
    "CANFINHOME": "NBFC", "LICHSGFIN": "NBFC", "PFC": "NBFC",
    "RECLTD": "NBFC", "IRFC": "NBFC", "SBILIFE": "Insurance",
    "ICICIPRULI": "Insurance", "SBICARD": "Insurance", "LICI": "Insurance",
    "JIOFIN": "NBFC",
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma", "AUROPHARMA": "Pharma", "LUPIN": "Pharma",
    "TORNTPHARM": "Pharma", "IPCALAB": "Pharma", "ZYDUSLIFE": "Pharma",
    "SYNGENE": "Pharma", "LALPATHLAB": "Pharma", "APOLLOHOSP": "Healthcare",
    "MAXHEALTH": "Healthcare", "FORTIS": "Healthcare",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto", "ESCORTS": "Auto", "MOTHERSON": "Auto",
    "BALKRISIND": "Auto",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "COLPAL": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "TATACONSUM": "FMCG",
    "TRENT": "FMCG", "UNITDSPR": "FMCG", "PAGEIND": "FMCG",
    "BATAINDIA": "FMCG",
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "PETRONET": "Energy",
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "ADANIGREEN": "Power", "ADANIPOWER": "Power", "NHPC": "Power",
    "JSWENERGY": "Power", "COALINDIA": "Mining",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "JINDALSTEL": "Metals",
    "LT": "Infra", "ADANIPORTS": "Infra", "CONCOR": "Logistics",
    "GMRINFRA": "Infra", "SIEMENS": "Industrial", "HAL": "Defence",
    "BEL": "Defence", "CUMMINSIND": "Industrial", "TIINDIA": "Industrial",
    "BOSCHLTD": "Industrial",
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "RAMCOCEM": "Cement", "ASTRAL": "Building", "PIDILITIND": "Chemicals",
    "DLF": "Realty", "OBEROIRLTY": "Realty", "LODHA": "Realty",
    "PHOENIXLTD": "Realty",
    "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom", "TATACOMM": "Telecom",
    "SRF": "Chemicals", "PIIND": "Chemicals", "DEEPAKNTR": "Chemicals",
    "ATUL": "Chemicals", "TATACHEM": "Chemicals",
    "POLYCAB": "Electricals", "HAVELLS": "Electricals", "VOLTAS": "Electricals",
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "BERGEPAINT": "Consumer",
    "ADANIENT": "Conglomerate", "DMART": "Retail", "JUBLFOOD": "QSR",
    "IRCTC": "Travel", "INDHOTEL": "Hotels", "PVRINOX": "Media",
    "ZOMATO": "Internet", "PAYTM": "Internet", "CDSL": "Exchange",
    "TRIDENT": "Textiles",
}


def get_universe(tier="nifty200"):
    if tier == "nifty50":
        return list(NIFTY50)
    elif tier == "nifty100":
        return list(NIFTY50) + list(NIFTY_NEXT50)
    else:
        return list(NIFTY50) + list(NIFTY_NEXT50) + list(MIDCAP_SELECT)


def fetch_universe_prices(symbols, years=12, end_date=None):
    """Batch-fetch via yfinance. Returns {symbol: DataFrame}."""
    import yfinance as yf
    end = end_date or datetime.now()
    start = end - timedelta(days=years * 365)
    yf_symbols = [f"{s}.NS" for s in symbols]
    yf_to_local = {f"{s}.NS": s for s in symbols}

    print(f"  Downloading {len(yf_symbols)} stocks...")
    try:
        raw = yf.download(yf_symbols, start=start, end=end, group_by="ticker",
                          auto_adjust=True, threads=True, progress=True)
    except Exception as e:
        print(f"  Download failed: {e}")
        return {}

    result, failed = {}, 0
    for yf_sym, local in yf_to_local.items():
        try:
            df = raw[yf_sym].copy() if len(yf_symbols) > 1 else raw.copy()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].dropna(subset=["close"])
            if len(df) >= 252:
                result[local] = df
            else:
                failed += 1
        except Exception:
            failed += 1
    print(f"  Loaded: {len(result)} | Failed: {failed}")
    return result


def load_fundamental_data(data_dir="data"):
    """Load Screener.in cache. Returns {symbol: dict}."""
    for name in ["fundamental_scan.json", "fundamental_full_universe.json"]:
        path = Path(data_dir) / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", [])
            fund = {r["symbol"]: r for r in results if r.get("symbol")}
            if fund:
                print(f"  Loaded fundamentals: {len(fund)} stocks")
                return fund
    print("  No fundamental data found")
    return {}

def get_sector_map():
    return dict(SECTOR_MAP)
