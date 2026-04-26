"""
Intraday signal loop V2 — uses Tier 1 watchlist.

Runs every 5 minutes during 9:15 - 15:30 IST.
Scans ~57 Tier 1 instruments (NIFTY 50 + MCX + Currency).
"""

from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles, compute_indicators
from core.regime import detect_regime
from core.meta_allocator import compute_composite
from core.risk_gate import evaluate_risk
from core.watchlist import WatchlistManager
from execution.orders import execute_signal
from execution.portfolio_tracker import open_position, update_positions
from execution.telegram_bot import send_signal
from utils.logger import log
from utils import is_market_hours, now_ist, load_json


SIGNAL_THRESHOLD = 0.4
_recent_signals = {}  # symbol -> last signal timestamp
COOLDOWN_HOURS = 4


def _is_on_cooldown(symbol: str) -> bool:
    from datetime import datetime, timedelta
    last = _recent_signals.get(symbol)
    if not last:
        return False
    return datetime.now() - last < timedelta(hours=COOLDOWN_HOURS)


def _mark_signaled(symbol: str):
    from datetime import datetime
    _recent_signals[symbol] = datetime.now()


def run():
    """Execute one intraday scan cycle on Tier 1."""
    if not is_market_hours():
        log.debug("Outside market hours, skipping intraday cycle")
        return

    # Check auth
    kite = get_kite()
    if kite is None:
        log.warning("Intraday cycle skipped — no valid Kite session")
        return

    log.info(f"─── Intraday cycle at {now_ist().strftime('%H:%M:%S')} ───")

    # Load Tier 1 watchlist
    mgr = WatchlistManager()
    tier1 = mgr.get_tier1()
    equities = [w for w in tier1 if w.get("type") in ("equity", "commodity")]

    if not equities:
        # Fallback to old watchlist
        equities = [w for w in cfg.WATCHLIST if w.get("type") == "equity"]

    signals_fired = 0

    for item in equities:
        symbol = item["symbol"]
        token = item.get("token", 0)
        exchange = item.get("exchange", "NSE")
        sector = item.get("sector", "Unknown")

        if not token:
            continue

        try:
            daily_df = fetch_candles(token, interval="day", days=120)
            if daily_df.empty or len(daily_df) < 60:
                continue

            daily_df = compute_indicators(daily_df)
            regime = detect_regime(daily_df)
            signal = compute_composite(daily_df, regime, signal_threshold=SIGNAL_THRESHOLD)

            if signal.direction == "HOLD":
                continue

            current_price = daily_df["close"].iloc[-1]
            atr = daily_df["atr"].iloc[-1] if "atr" in daily_df.columns else current_price * 0.02

            # Determine product type
            if item.get("type") == "commodity":
                product = "NRML"
            else:
                product = "MIS"

            risk = evaluate_risk(
                signal=signal,
                symbol=symbol,
                sector=sector,
                current_price=current_price,
                atr=atr,
                product=product,
            )

            if not risk.approved:
                log.debug(f"{symbol}: Signal {signal.signal:+.3f} rejected — {risk.reason}")
                continue

            if _is_on_cooldown(symbol):
                log.debug(f"{symbol}: On cooldown, skipping")
                continue

            _mark_signaled(symbol)
            signals_fired += 1

            order = execute_signal(
                symbol=symbol,
                exchange=exchange,
                signal=signal,
                risk=risk,
                product=product,
            )

            # Track in paper portfolio
            if cfg.TRADING_MODE == "paper":
                open_position(
                    symbol=symbol,
                    direction=signal.direction,
                    quantity=risk.position_size,
                    entry_price=current_price,
                    stop_loss=risk.stop_loss,
                    target=risk.target,
                    regime=regime.regime,
                    signal_strength=signal.signal,
                )

            if cfg.TRADING_MODE == "approval" and order.get("status") == "pending_approval":
                send_signal(order)

        except Exception as e:
            log.warning(f"Intraday scan failed for {symbol}: {e}")

    # Update open positions with current prices
    try:
        price_updates = {}
        for item in equities:
            token = item.get("token", 0)
            if token:
                try:
                    df_temp = fetch_candles(token, interval="day", days=5)
                    if not df_temp.empty:
                        price_updates[item["symbol"]] = df_temp["close"].iloc[-1]
                except Exception:
                    pass
        if price_updates:
            update_positions(price_updates)
    except Exception as e:
        log.debug(f"Position update failed: {e}")

    log.info(f"─── Intraday cycle complete: {signals_fired} signals from {len(equities)} instruments ───")


if __name__ == "__main__":
    run()
