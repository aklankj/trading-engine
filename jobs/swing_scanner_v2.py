"""
Swing Scanner V2 — single signal source for the engine.

Uses strategies.registry for all signals. No other signal source exists.
The old intraday loop, meta_composite, and core/strategies.py are dead.

Runs once daily at 10:00 AM. Also updates open positions at 2:30 PM.
"""

from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles
from core.watchlist import WatchlistManager
from strategies.registry import get_composite_signal, get_all_signals
from execution.portfolio_tracker import (
    open_position, update_positions, _load_portfolio,
)
from execution.telegram_bot import send_message
from utils.logger import log
from utils import now_ist, load_json


MAX_POSITIONS = 10
MAX_NEW_PER_DAY = 3
MIN_SIGNAL_STRENGTH = 0.5


def run():
    """Daily swing signal scan."""
    current = now_ist()

    if current.hour < 9 or current.hour > 16:
        log.debug("Outside market hours, skipping swing scan")
        return

    kite = get_kite()
    if kite is None:
        log.warning("Swing scan skipped — no Kite session")
        return

    log.info(f"═══ SWING SCAN V2 — {current.strftime('%Y-%m-%d %H:%M')} ═══")

    # Load watchlist
    mgr = WatchlistManager()
    tier1 = mgr.get_tier1()
    tier2 = mgr.get_tier2()
    equities = [w for w in tier1 + tier2 if w.get("type") == "equity"]

    portfolio = _load_portfolio()
    existing = set(portfolio.get("positions", {}).keys())

    signals_found = []
    price_updates = {}

    for item in equities:
        symbol = item["symbol"]
        token = item.get("token", 0)
        if not token:
            continue

        try:
            df = fetch_candles(token, interval="day", days=500)
            if df.empty or len(df) < 200:
                continue

            current_price = float(df["close"].iloc[-1])
            price_updates[symbol] = current_price

            if symbol in existing:
                continue

            # Get composite signal from registry
            composite = get_composite_signal(df)

            if abs(composite.signal) > MIN_SIGNAL_STRENGTH:
                # Validate SL/TGT are never zero
                if composite.stop_loss <= 0 or composite.target <= 0:
                    log.debug(f"  {symbol}: skipped — invalid SL/TGT")
                    continue

                signals_found.append({
                    "symbol": symbol,
                    "sector": item.get("sector", "Unknown"),
                    "tier": item.get("tier", 0),
                    "signal": composite,
                    "price": current_price,
                })

                log.info(
                    f"  {composite.direction} {symbol}: signal={composite.signal:+.2f} "
                    f"| {composite.strategy} | {composite.reason}"
                )

            import time
            time.sleep(0.4)

        except Exception as e:
            log.debug(f"  {symbol}: {e}")

    # Update existing positions
    if price_updates:
        update_positions(price_updates)

    # Open new positions (max 3 per day, max 10 total)
    new_opened = 0
    signals_found.sort(key=lambda x: abs(x["signal"].signal), reverse=True)

    for sig_data in signals_found:
        if new_opened >= MAX_NEW_PER_DAY:
            break

        portfolio = _load_portfolio()
        if len(portfolio.get("positions", {})) >= MAX_POSITIONS:
            log.info("Max positions reached")
            break

        sig = sig_data["signal"]
        symbol = sig_data["symbol"]
        price = sig_data["price"]

        # Position sizing
        equity = portfolio["cash"]
        for pos in portfolio["positions"].values():
            equity += float(pos.get("current_price", pos["entry_price"])) * int(pos["quantity"])

        alloc = min(0.10, 0.05 + (abs(sig.signal) - 0.5) * 0.10)
        quantity = max(1, int(equity * alloc / price))

        # Finding #9: Reject SELL for cash equities (not shortable in India)
        if sig.direction == "SELL":
            log.debug(f"  {symbol}: SELL rejected — cash equities not shortable")
            continue

        # Validate SL/TGT are sensible
        if sig.stop_loss <= 0 or sig.target <= 0:
            log.debug(f"  {symbol}: skipped — invalid SL={sig.stop_loss} TGT={sig.target}")
            continue

        open_position(
            symbol=symbol,
            direction=sig.direction,
            quantity=quantity,
            entry_price=price,
            stop_loss=sig.stop_loss,
            target=sig.target,
            regime="",
            signal_strength=sig.signal,
            strategy=sig.strategy,
        )

        # Set max hold days based on strategy
        portfolio_after = _load_portfolio()
        if symbol in portfolio_after.get("positions", {}):
            portfolio_after["positions"][symbol]["max_hold_days"] = sig.hold_days or 90
            from execution.portfolio_tracker import _save_portfolio
            _save_portfolio(portfolio_after)
        new_opened += 1

    log.info(
        f"═══ SWING SCAN COMPLETE: {len(equities)} scanned, "
        f"{len(signals_found)} signals, {new_opened} new positions ═══"
    )

    # Telegram
    if signals_found:
        _send_signals_telegram(signals_found, new_opened, len(equities))


def update_open_positions():
    """Update all open positions with current prices. Runs at 2:30 PM."""
    kite = get_kite()
    if kite is None:
        return

    log.info("Updating open positions...")

    portfolio = _load_portfolio()
    if not portfolio.get("positions"):
        return

    cache = load_json(cfg.DATA_DIR / "instrument_cache.json", default={})
    nse_tokens = {i["symbol"]: i["token"] for i in cache.get("nse", [])}

    prices = {}
    for sym in portfolio["positions"]:
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
        log.info(f"Updated {len(prices)} positions")


def _send_signals_telegram(signals, new_opened, total_scanned):
    """Send swing signals to Telegram."""
    try:
        current = now_ist()
        msg = (
            f"🔔 <b>SWING SIGNALS — {current.strftime('%Y-%m-%d')}</b>\n\n"
            f"Scanned: {total_scanned} stocks\n"
            f"Signals: {len(signals)} | New positions: {new_opened}\n\n"
        )

        for sig_data in signals[:8]:
            sig = sig_data["signal"]
            emoji = "🟢" if sig.direction == "BUY" else "🔴"
            msg += (
                f"{emoji} <b>{sig.direction} {sig_data['symbol']}</b> "
                f"@ ₹{sig_data['price']:,.0f}\n"
                f"  Signal: {sig.signal:+.2f} | {sig.strategy}\n"
                f"  {sig.reason}\n"
                f"  SL: ₹{sig.stop_loss:,.0f} | TGT: ₹{sig.target:,.0f}\n\n"
            )

        send_message(msg)
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    run()
