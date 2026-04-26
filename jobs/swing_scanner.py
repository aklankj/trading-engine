"""
Swing Signal Scanner — replaces the old 5-minute intraday loop.

Runs ONCE per day at 10:00 AM (after market settles).
Scans all Tier 1 + Tier 2 instruments with swing strategies.
Opens/manages positions in the paper portfolio.

Why once per day:
- Swing strategies use daily/weekly candles — no point checking every 5 min
- Reduces Kite API load from ~700 calls/hour to ~160 calls/day
- Fewer, higher-conviction signals instead of noise
"""

import time
from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles
from core.swing_strategies import get_all_signals, get_composite_signal, SWING_STRATEGIES
from core.watchlist import WatchlistManager
from execution.portfolio_tracker import (
    open_position, update_positions, get_daily_summary,
    _load_portfolio,
)
from execution.telegram_bot import send_message
from utils.logger import log
from utils import now_ist


def run():
    """Execute daily swing signal scan."""
    current = now_ist()

    # Only run during market hours
    if current.hour < 9 or current.hour > 16:
        log.debug("Outside market hours, skipping swing scan")
        return

    kite = get_kite()
    if kite is None:
        log.warning("Swing scan skipped — no Kite session")
        return

    log.info(f"═══ SWING SIGNAL SCAN — {current.strftime('%Y-%m-%d %H:%M')} ═══")

    # Load watchlist
    mgr = WatchlistManager()
    tier1 = mgr.get_tier1()
    tier2 = mgr.get_tier2()

    # Combine Tier 1 + Tier 2 equities
    all_equities = [w for w in tier1 + tier2 if w.get("type") in ("equity",)]

    portfolio = _load_portfolio()
    existing_positions = set(portfolio.get("positions", {}).keys())

    signals_found = []
    price_updates = {}

    for item in all_equities:
        symbol = item["symbol"]
        token = item.get("token", 0)

        if not token:
            continue

        try:
            # Fetch 2 years of daily data for strategy calculations
            df = fetch_candles(token, interval="day", days=500)
            if df.empty or len(df) < 200:
                continue

            current_price = df["close"].iloc[-1]
            price_updates[symbol] = current_price

            # Skip if already have position
            if symbol in existing_positions:
                continue

            # Get composite signal from all 6 strategies
            composite = get_composite_signal(df)

            if abs(composite.signal) > 0.5:  # Higher threshold for swing trades
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

            time.sleep(0.4)  # Rate limit

        except Exception as e:
            log.debug(f"  {symbol}: {e}")

    # Update existing positions with current prices
    if price_updates:
        update_positions(price_updates)

    # Open new positions for strong signals (max 3 new per day)
    new_positions = 0
    max_new = 3

    # Sort by absolute signal strength
    signals_found.sort(key=lambda x: abs(x["signal"].signal), reverse=True)

    for sig_data in signals_found:
        if new_positions >= max_new:
            break

        sig = sig_data["signal"]
        symbol = sig_data["symbol"]
        price = sig_data["price"]

        # Check position limits
        portfolio = _load_portfolio()
        if len(portfolio.get("positions", {})) >= 10:
            log.info("Max 10 positions reached, skipping new entries")
            break

        # Calculate position size (5-10% of equity based on conviction)
        equity = portfolio["cash"]
        for pos in portfolio["positions"].values():
            equity += pos.get("current_price", pos["entry_price"]) * pos["quantity"]

        alloc_pct = 0.05 + (abs(sig.signal) - 0.5) * 0.10  # 5-10%
        alloc_pct = min(0.10, alloc_pct)
        position_value = equity * alloc_pct
        quantity = max(1, int(position_value / price))

        # Use strategy's stop/target, or default
        stop_loss = sig.stop_loss if sig.stop_loss > 0 else price * 0.90
        target = sig.target if sig.target > 0 else price * 1.20

        open_position(
            symbol=symbol,
            direction=sig.direction,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_loss,
            target=target,
            regime=sig_data.get("regime", "Unknown"),
            signal_strength=sig.signal,
            strategy=sig.strategy,
        )

        new_positions += 1

    # Summary
    log.info(
        f"═══ SWING SCAN COMPLETE ═══\n"
        f"  Scanned: {len(all_equities)} stocks\n"
        f"  Signals found: {len(signals_found)}\n"
        f"  New positions opened: {new_positions}\n"
        f"  Existing positions updated: {len(existing_positions)}"
    )

    # Telegram alert for new positions
    if signals_found:
        msg = f"🔔 <b>SWING SIGNALS — {current.strftime('%Y-%m-%d')}</b>\n\n"
        msg += f"Scanned: {len(all_equities)} stocks\n"
        msg += f"Signals: {len(signals_found)} | New positions: {new_positions}\n\n"

        for sig_data in signals_found[:8]:
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


if __name__ == "__main__":
    run()
