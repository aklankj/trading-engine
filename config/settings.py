"""
Central configuration — loads .env and provides typed constants.
Import this everywhere: from config.settings import cfg
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "config" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv(BASE_DIR / ".env")


class _Config:
    """Singleton configuration object."""

    # ── Kite Connect ──────────────────────────────────────────
    KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
    KITE_API_SECRET: str = os.getenv("KITE_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")
    KITE_USER_ID: str = os.getenv("KITE_USER_ID", "")
    KITE_PASSWORD: str = os.getenv("KITE_PASSWORD", "")
    KITE_TOTP_SECRET: str = os.getenv("KITE_TOTP_SECRET", "")

    # ── Telegram ──────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── OpenRouter ────────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    # ── Capital & Risk ────────────────────────────────────────
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
    DEBUG_SLEEP: bool = False  # Enable rate-limit sleeps (off by default)
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    MAX_SECTOR_EXPOSURE_PCT: float = float(os.getenv("MAX_SECTOR_EXPOSURE_PCT", "0.30"))
    FUNDAMENTAL_ALLOC: float = float(os.getenv("FUNDAMENTAL_ALLOCATION_PCT", "0.60"))
    ACTIVE_ALLOC: float = float(os.getenv("ACTIVE_ALLOCATION_PCT", "0.30"))
    CASH_RESERVE: float = float(os.getenv("CASH_RESERVE_PCT", "0.10"))

    # ── Paths ─────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")
    TRADE_LOG: Path = BASE_DIR / os.getenv("TRADE_LOG", "data/trade_log.json")
    PAPER_LOG: Path = BASE_DIR / os.getenv("PAPER_LOG", "data/paper_log.json")
    SIGNAL_LOG: Path = BASE_DIR / os.getenv("SIGNAL_LOG", "data/signal_log.csv")

    # ── Mode ──────────────────────────────────────────────────
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")  # paper | approval | semi_auto

    # ── Watchlist ─────────────────────────────────────────────
    _watchlist_path: Path = BASE_DIR / "config" / "watchlist.json"

    @property
    def WATCHLIST(self) -> list[dict]:
        if self._watchlist_path.exists():
            return json.loads(self._watchlist_path.read_text())
        return self._default_watchlist()

    @staticmethod
    def _default_watchlist() -> list[dict]:
        return [
            {"symbol": "NIFTY 50", "token": 256265, "exchange": "NSE", "sector": "Index", "type": "index"},
            {"symbol": "NIFTY BANK", "token": 260105, "exchange": "NSE", "sector": "Index", "type": "index"},
            {"symbol": "RELIANCE", "token": 738561, "exchange": "NSE", "sector": "Energy", "type": "equity"},
            {"symbol": "HDFCBANK", "token": 341249, "exchange": "NSE", "sector": "Banking", "type": "equity"},
            {"symbol": "INFY", "token": 408065, "exchange": "NSE", "sector": "IT", "type": "equity"},
            {"symbol": "TCS", "token": 2953217, "exchange": "NSE", "sector": "IT", "type": "equity"},
            {"symbol": "ICICIBANK", "token": 1270529, "exchange": "NSE", "sector": "Banking", "type": "equity"},
            {"symbol": "BAJFINANCE", "token": 81153, "exchange": "NSE", "sector": "NBFC", "type": "equity"},
            {"symbol": "ASIANPAINT", "token": 60417, "exchange": "NSE", "sector": "Consumer", "type": "equity"},
            {"symbol": "TITAN", "token": 897537, "exchange": "NSE", "sector": "Consumer", "type": "equity"},
            {"symbol": "PIDILITIND", "token": 681985, "exchange": "NSE", "sector": "Chemicals", "type": "equity"},
            {"symbol": "SBIN", "token": 779521, "exchange": "NSE", "sector": "Banking", "type": "equity"},
            {"symbol": "BHARTIARTL", "token": 2714625, "exchange": "NSE", "sector": "Telecom", "type": "equity"},
            {"symbol": "ITC", "token": 424961, "exchange": "NSE", "sector": "FMCG", "type": "equity"},
            {"symbol": "MARUTI", "token": 2815745, "exchange": "NSE", "sector": "Auto", "type": "equity"},
            {"symbol": "LT", "token": 2939649, "exchange": "NSE", "sector": "Infra", "type": "equity"},
            {"symbol": "AXISBANK", "token": 1510401, "exchange": "NSE", "sector": "Banking", "type": "equity"},
            {"symbol": "KOTAKBANK", "token": 492033, "exchange": "NSE", "sector": "Banking", "type": "equity"},
            {"symbol": "SUNPHARMA", "token": 857857, "exchange": "NSE", "sector": "Pharma", "type": "equity"},
            {"symbol": "WIPRO", "token": 969473, "exchange": "NSE", "sector": "IT", "type": "equity"},
        ]

    def ensure_dirs(self):
        """Create data and log directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return list of missing critical configs."""
        issues = []
        if not self.KITE_API_KEY:
            issues.append("KITE_API_KEY not set")
        if not self.KITE_API_SECRET:
            issues.append("KITE_API_SECRET not set")
        if not self.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN not set")
        if not self.TELEGRAM_CHAT_ID:
            issues.append("TELEGRAM_CHAT_ID not set")
        return issues


cfg = _Config()
cfg.ensure_dirs()
