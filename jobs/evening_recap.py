"""
Evening recap V2 — uses portfolio tracker for real P&L.
Runs at 4:00 PM IST.
"""

from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles
from execution.portfolio_tracker import (
    _load_portfolio, update_positions, get_daily_summary, format_daily_telegram
)
from execution.telegram_bot import send_message
from utils.logger import log
from utils import load_json


def run():
    """Generate and send daily recap with real P&L."""
    log.info("═══ EVENING RECAP ═══")

    # Update all positions with closing prices
    try:
        kite = get_kite()
        if kite:
            portfolio = _load_portfolio()
            cache = load_json(cfg.DATA_DIR / "instrument_cache.json", default={})
            nse_tokens = {i["symbol"]: i["token"] for i in cache.get("nse", [])}

            prices = {}
            for sym in portfolio.get("positions", {}):
                token = nse_tokens.get(sym, 0)
                if token:
                    try:
                        df = fetch_candles(token, interval="day", days=5)
                        if not df.empty:
                            prices[sym] = float(df["close"].iloc[-1])
                    except Exception:
                        pass

            if prices:
                update_positions(prices)
    except Exception as e:
        log.warning(f"Position update failed: {e}")

    # Generate and send summary
    summary = get_daily_summary()
    text = format_daily_telegram(summary)
    send_message(text)

    log.info(
        f"Daily recap sent | "
        f"Equity: {summary['equity']} | "
        f"P&L: {summary['return_pct']:+.1f}% | "
        f"Open: {summary['positions_open']} | "
        f"Trades: {summary['total_trades']}"
    )


if __name__ == "__main__":
    run()
