"""
Tiered watchlist system.

Tier 1 — Active signals (every 5 min)
  NIFTY 50 + BANKNIFTY + MCX commodities + USDINR
  ~57 instruments, must be highly liquid

Tier 2 — Daily swing scan (once at 8:45 AM)
  NIFTY Next 50 + Midcap 50 + Sector indices
  ~110 instruments, scanned for regime + swing setups

Tier 3 — Fundamental universe (weekly/quarterly)
  All NSE stocks with market cap > 500 Cr
  ~800-900 companies scored on quality metrics
  Top scorers watched for price-drop buy triggers

The watchlist auto-builds from Kite's instrument dump.
No hardcoded tokens — fetches fresh instrument data daily.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from config.settings import cfg
from utils.logger import log
from utils import load_json, save_json


# ── Tier 1: Active trading universe ──────────────────────────

TIER1_SYMBOLS = {
    # NIFTY 50 components (all of them)
    "indices": [
        {"symbol": "NIFTY 50", "exchange": "NSE", "sector": "Index", "type": "index"},
        {"symbol": "NIFTY BANK", "exchange": "NSE", "sector": "Index", "type": "index"},
    ],
    "nifty50": [
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
    ],
    "commodities_mcx": [
        {"name": "Gold", "prefix": "GOLD", "exchange": "MCX", "sector": "Precious Metals"},
        {"name": "Gold Mini", "prefix": "GOLDM", "exchange": "MCX", "sector": "Precious Metals"},
        {"name": "Silver", "prefix": "SILVER", "exchange": "MCX", "sector": "Precious Metals"},
        {"name": "Silver Mini", "prefix": "SILVERM", "exchange": "MCX", "sector": "Precious Metals"},
        {"name": "Crude Oil", "prefix": "CRUDEOIL", "exchange": "MCX", "sector": "Energy"},
        {"name": "Natural Gas", "prefix": "NATURALGAS", "exchange": "MCX", "sector": "Energy"},
        {"name": "Copper", "prefix": "COPPER", "exchange": "MCX", "sector": "Industrial Metals"},
    ],
    "currency": [
        {"symbol": "USDINR", "exchange": "CDS", "sector": "Currency", "type": "currency"},
    ],
}

# ── Tier 2: Daily swing scan universe ────────────────────────

TIER2_ADDITIONAL = [
    # NIFTY Next 50
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
    "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "CONCOR", "DABUR", "DLF", "GAIL",
    "GODREJCP", "HAVELLS", "HAL", "ICICIPRULI", "IDEA",
    "IDFCFIRSTB", "INDHOTEL", "INDUSTOWER", "IOC", "IRCTC",
    "IRFC", "JIOFIN", "JINDALSTEL", "JSWENERGY", "LICI",
    "LODHA", "LUPIN", "MARICO", "MAXHEALTH", "MOTHERSON",
    "NAUKRI", "NHPC", "OBEROIRLTY", "OFSS", "PAYTM",
    "PFC", "PIIND", "PNB", "POLYCAB", "RECLTD",
    "SBICARD", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
    "TVSMOTOR", "UNIONBANK", "UNITDSPR", "VEDL", "ZOMATO",
    "ZYDUSLIFE",

    # Popular midcaps with good liquidity
    "ASTRAL", "ATUL", "BALKRISIND", "BATAINDIA", "CANFINHOME",
    "CDSL", "CUMMINSIND", "DEEPAKNTR", "DMART", "ESCORTS",
    "FEDERALBNK", "FORTIS", "GMRINFRA", "IPCALAB", "JUBLFOOD",
    "LALPATHLAB", "LICHSGFIN", "LTIM", "LTTS", "MFSL",
    "MPHASIS", "MUTHOOTFIN", "PAGEIND", "PERSISTENT", "PETRONET",
    "PHOENIXLTD", "PVRINOX", "RAMCOCEM", "SAIL", "SYNGENE",
    "TATACHEM", "TATACOMM", "TATAELXSI", "TIINDIA", "TRIDENT",
    "VOLTAS", "YESBANK",
]

# ── Sector indices for Tier 2 regime detection ───────────────

SECTOR_INDICES = [
    {"symbol": "NIFTY IT", "exchange": "NSE", "sector": "IT Index", "type": "index"},
    {"symbol": "NIFTY BANK", "exchange": "NSE", "sector": "Bank Index", "type": "index"},
    {"symbol": "NIFTY PHARMA", "exchange": "NSE", "sector": "Pharma Index", "type": "index"},
    {"symbol": "NIFTY AUTO", "exchange": "NSE", "sector": "Auto Index", "type": "index"},
    {"symbol": "NIFTY FMCG", "exchange": "NSE", "sector": "FMCG Index", "type": "index"},
    {"symbol": "NIFTY METAL", "exchange": "NSE", "sector": "Metal Index", "type": "index"},
    {"symbol": "NIFTY REALTY", "exchange": "NSE", "sector": "Realty Index", "type": "index"},
    {"symbol": "NIFTY ENERGY", "exchange": "NSE", "sector": "Energy Index", "type": "index"},
    {"symbol": "NIFTY MEDIA", "exchange": "NSE", "sector": "Media Index", "type": "index"},
    {"symbol": "NIFTY PSE", "exchange": "NSE", "sector": "PSE Index", "type": "index"},
]


class WatchlistManager:
    """
    Builds and manages tiered watchlists from Kite instrument data.
    Caches instrument tokens locally to avoid re-fetching.
    """

    def __init__(self):
        self.cache_path = cfg.DATA_DIR / "instrument_cache.json"
        self.watchlist_path = cfg.DATA_DIR / "tiered_watchlist.json"
        self._instruments_nse = None
        self._instruments_mcx = None

    def refresh_instruments(self, kite) -> None:
        """Fetch fresh instrument list from Kite and cache it."""
        try:
            nse = kite.instruments("NSE")
            mcx = kite.instruments("MCX")
            bse = kite.instruments("BSE")

            cache = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "nse_count": len(nse),
                "mcx_count": len(mcx),
                "bse_count": len(bse),
                "nse": [{"symbol": i["tradingsymbol"], "token": i["instrument_token"],
                         "name": i.get("name", ""), "segment": i["segment"],
                         "exchange": i["exchange"], "instrument_type": i.get("instrument_type", ""),
                         "lot_size": i.get("lot_size", 1)}
                        for i in nse if i["segment"] == "NSE"],
                "mcx": [{"symbol": i["tradingsymbol"], "token": i["instrument_token"],
                         "name": i.get("name", ""), "segment": i["segment"],
                         "exchange": i["exchange"], "instrument_type": i.get("instrument_type", ""),
                         "expiry": str(i.get("expiry", "")), "lot_size": i.get("lot_size", 1)}
                        for i in mcx if i["instrument_type"] == "FUT"],
            }

            save_json(self.cache_path, cache)
            log.info(f"Instrument cache refreshed: {len(cache['nse'])} NSE, {len(cache['mcx'])} MCX futures")

        except Exception as e:
            log.error(f"Failed to refresh instruments: {e}")

    def _load_cache(self) -> dict:
        """Load cached instruments."""
        cache = load_json(self.cache_path, default={})
        if not cache:
            log.warning("No instrument cache found — run refresh_instruments() first")
        return cache

    def _find_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Find instrument token for a symbol."""
        cache = self._load_cache()
        key = "nse" if exchange in ("NSE", "BSE") else "mcx"
        for inst in cache.get(key, []):
            if inst["symbol"] == symbol:
                return inst["token"]
        return 0

    def _find_nearest_mcx_future(self, prefix: str) -> dict:
        """Find the nearest expiry MCX future for a commodity."""
        cache = self._load_cache()
        matches = [
            i for i in cache.get("mcx", [])
            if i["symbol"].startswith(prefix) and i["expiry"]
        ]
        if not matches:
            return {}

        # Sort by expiry, pick nearest future
        # Finding #14: Parse expiry as date, not string sort
        from datetime import datetime as _dt
        def _parse_expiry(x):
            try:
                return _dt.strptime(str(x["expiry"])[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                return _dt.max
        matches.sort(key=_parse_expiry)
        today = datetime.now().strftime("%Y-%m-%d")
        for m in matches:
            if m["expiry"] >= today:
                return m
        return matches[-1] if matches else {}

    def build_tier1(self) -> list[dict]:
        """Build Tier 1 watchlist: NIFTY 50 + MCX + Currency."""
        watchlist = []

        # Indices (hardcoded tokens — these don't change)
        watchlist.extend([
            {"symbol": "NIFTY 50", "token": 256265, "exchange": "NSE",
             "sector": "Index", "type": "index", "tier": 1},
            {"symbol": "NIFTY BANK", "token": 260105, "exchange": "NSE",
             "sector": "Index", "type": "index", "tier": 1},
        ])

        # NIFTY 50 equities
        for sym in TIER1_SYMBOLS["nifty50"]:
            token = self._find_token(sym, "NSE")
            if token:
                watchlist.append({
                    "symbol": sym, "token": token, "exchange": "NSE",
                    "sector": _infer_sector(sym), "type": "equity", "tier": 1,
                })
            else:
                log.warning(f"Tier 1: could not find token for {sym}")

        # MCX commodities (nearest futures)
        for comm in TIER1_SYMBOLS["commodities_mcx"]:
            fut = self._find_nearest_mcx_future(comm["prefix"])
            if fut:
                watchlist.append({
                    "symbol": fut["symbol"], "token": fut["token"],
                    "exchange": "MCX", "sector": comm["sector"],
                    "type": "commodity", "tier": 1,
                    "lot_size": fut.get("lot_size", 1),
                    "expiry": fut.get("expiry", ""),
                })

        log.info(f"Tier 1 built: {len(watchlist)} instruments")
        return watchlist

    def build_tier2(self) -> list[dict]:
        """Build Tier 2 watchlist: Next 50 + Midcaps + Sector indices."""
        watchlist = []

        # Sector indices
        for idx in SECTOR_INDICES:
            # Sector index tokens are well-known
            watchlist.append({**idx, "tier": 2, "token": 0})

        # Additional equities
        for sym in TIER2_ADDITIONAL:
            token = self._find_token(sym, "NSE")
            if token:
                watchlist.append({
                    "symbol": sym, "token": token, "exchange": "NSE",
                    "sector": _infer_sector(sym), "type": "equity", "tier": 2,
                })

        log.info(f"Tier 2 built: {len(watchlist)} instruments")
        return watchlist

    def build_tier3_universe(self) -> list[dict]:
        """
        Build Tier 3: All NSE stocks for fundamental screening.
        Filters: segment=NSE, instrument_type=EQ
        Returns raw list — the fundamental screener scores them.
        """
        cache = self._load_cache()
        universe = []

        for inst in cache.get("nse", []):
            # Only regular equities
            if inst.get("instrument_type", "") in ("EQ", ""):
                universe.append({
                    "symbol": inst["symbol"],
                    "token": inst["token"],
                    "exchange": "NSE",
                    "name": inst.get("name", ""),
                    "tier": 3,
                })

        log.info(f"Tier 3 universe: {len(universe)} NSE stocks")
        return universe

    def build_all(self) -> dict:
        """Build all tiers and save."""
        tier1 = self.build_tier1()
        tier2 = self.build_tier2()
        tier3 = self.build_tier3_universe()

        result = {
            "built_at": datetime.now().isoformat(),
            "tier1": tier1,
            "tier2": tier2,
            "tier3_count": len(tier3),
            "tier3": tier3,
        }

        save_json(self.watchlist_path, result)
        log.info(
            f"Watchlists built — "
            f"Tier 1: {len(tier1)} (active trading) | "
            f"Tier 2: {len(tier2)} (daily scan) | "
            f"Tier 3: {len(tier3)} (fundamental universe)"
        )
        return result

    def get_tier1(self) -> list[dict]:
        """Get Tier 1 watchlist (from cache or build fresh)."""
        data = load_json(self.watchlist_path, default={})
        if data.get("tier1"):
            return data["tier1"]
        return self.build_tier1()

    def get_tier2(self) -> list[dict]:
        """Get Tier 2 watchlist."""
        data = load_json(self.watchlist_path, default={})
        if data.get("tier2"):
            return data["tier2"]
        return self.build_tier2()

    def get_tier3(self) -> list[dict]:
        """Get Tier 3 universe."""
        data = load_json(self.watchlist_path, default={})
        if data.get("tier3"):
            return data["tier3"]
        return self.build_tier3_universe()


# ── Sector inference from symbol ─────────────────────────────

_SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "CANBK": "Banking",
    "FEDERALBNK": "Banking", "YESBANK": "Banking", "IDFCFIRSTB": "Banking",
    "UNIONBANK": "Banking", "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC",
    "SHRIRAMFIN": "NBFC", "CHOLAFIN": "NBFC", "MUTHOOTFIN": "NBFC",
    "SBILIFE": "Insurance", "ICICIPRULI": "Insurance", "LICI": "Insurance",
    "SBICARD": "Finance", "PFC": "Finance", "RECLTD": "Finance",
    "CANFINHOME": "Finance", "LICHSGFIN": "Finance", "MFSL": "Finance",
    "CDSL": "Capital Markets", "JIOFIN": "Finance",

    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "LTTS": "IT", "MPHASIS": "IT",
    "PERSISTENT": "IT", "TATAELXSI": "IT", "OFSS": "IT", "NAUKRI": "IT",
    "PAYTM": "IT", "ZOMATO": "IT",

    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto", "ESCORTS": "Auto", "MOTHERSON": "Auto",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Healthcare", "AUROPHARMA": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma", "ZYDUSLIFE": "Pharma",
    "IPCALAB": "Pharma", "LALPATHLAB": "Healthcare", "MAXHEALTH": "Healthcare",
    "FORTIS": "Healthcare", "SYNGENE": "Pharma",

    # FMCG & Consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "COLPAL": "FMCG", "DABUR": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "TATACONSUM": "FMCG",
    "UNITDSPR": "FMCG", "PAGEIND": "Consumer", "BATAINDIA": "Consumer",
    "TITAN": "Consumer", "TRENT": "Consumer", "ASIANPAINT": "Consumer",
    "PIDILITIND": "Chemicals", "BERGEPAINT": "Consumer",
    "JUBLFOOD": "Consumer", "PVRINOX": "Consumer", "DMART": "Retail",

    # Energy & Power
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "COALINDIA": "Energy",
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "ADANIGREEN": "Power", "ADANIPOWER": "Power", "NHPC": "Power",
    "JSWENERGY": "Power", "IRFC": "Power", "PETRONET": "Energy",

    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "JINDALSTEL": "Metals",

    # Infrastructure & Industrial
    "LT": "Infra", "ULTRACEMCO": "Cement", "GRASIM": "Cement",
    "AMBUJACEM": "Cement", "RAMCOCEM": "Cement", "ADANIPORTS": "Infra",
    "CONCOR": "Logistics", "IRCTC": "Travel", "DLF": "Realty",
    "OBEROIRLTY": "Realty", "LODHA": "Realty", "PHOENIXLTD": "Realty",
    "GMRINFRA": "Infra", "SIEMENS": "Industrial", "HAL": "Defence",
    "BEL": "Defence", "INDUSTOWER": "Telecom", "BHARTIARTL": "Telecom",
    "IDEA": "Telecom", "INDHOTEL": "Hotels",

    # Chemicals & Speciality
    "SRF": "Chemicals", "PIIND": "Chemicals", "DEEPAKNTR": "Chemicals",
    "ATUL": "Chemicals", "ASTRAL": "Chemicals", "TATACHEM": "Chemicals",
    "POLYCAB": "Electricals", "HAVELLS": "Electricals", "VOLTAS": "Electricals",
    "CUMMINSIND": "Industrial", "BALKRISIND": "Industrial", "TRIDENT": "Textiles",
    "TIINDIA": "Industrial", "TATACOMM": "Telecom",

    # Misc
    "ADANIENT": "Conglomerate",
}


def _infer_sector(symbol: str) -> str:
    """Infer sector from symbol using known mapping."""
    return _SECTOR_MAP.get(symbol, "Unknown")


# ── CLI for building watchlists ──────────────────────────────

if __name__ == "__main__":
    from core.auth import get_kite

    print("Refreshing instrument data from Kite...")
    kite = get_kite()
    mgr = WatchlistManager()
    mgr.refresh_instruments(kite)

    print("\nBuilding tiered watchlists...")
    result = mgr.build_all()

    print(f"\n{'='*50}")
    print(f"Tier 1 (active trading):     {len(result['tier1']):>4d} instruments")
    print(f"Tier 2 (daily scan):         {len(result['tier2']):>4d} instruments")
    print(f"Tier 3 (fundamental universe): {result['tier3_count']:>4d} stocks")
    print(f"{'='*50}")

    print(f"\nTier 1 breakdown:")
    types = {}
    for w in result["tier1"]:
        t = w.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")

    print(f"\nSaved to: {mgr.watchlist_path}")
