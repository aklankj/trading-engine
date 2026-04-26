"""
Intraday signal loop — runs every 5 minutes during 9:15 - 15:30 IST.

For each watchlist equity:
1. Fetch latest 5-min candles
2. Detect regime
3. Run strategies → meta-allocator → composite signal
4. If signal strong enough, run through risk gate
5. Execute (paper/approval/semi-auto based on mode)
"""

from config.settings import cfg
from core.data import fetch_candles, compute_indicators
from core.regime import detect_regime
from core.meta_allocator import compute_composite
from core.risk_gate import evaluate_risk
from execution.orders import execute_signal
from execution.telegram_bot import send_signal
from utils.logger import log
from utils import is_market_hours, now_ist


# Only process equities, not indices (can't trade indices directly)
SIGNAL_THRESHOLD = 0.4  # Minimum composite signal to consider acting


def run():
    """Execute one intraday scan cycle."""
    if not is_market_hours():
        log.debug("Outside market hours, skipping intraday cycle")
        return

    log.info(f"─── Intraday cycle at {now_ist().strftime('%H:%M:%S')} ───")
    signals_fired = 0

    equities = [w for w in cfg.WATCHLIST if w.get("type") == "equity"]

    for item in equities:
        symbol = item["symbol"]
        token = item["token"]
        exchange = item.get("exchange", "NSE")
        sector = item.get("sector", "Unknown")

        try:
            # Fetch recent data (daily for regime, 5min for intraday signals)
            daily_df = fetch_candles(token, interval="day", days=120)
            if daily_df.empty or len(daily_df) < 60:
                continue

            daily_df = compute_indicators(daily_df)

            # Detect regime from daily data
            regime = detect_regime(daily_df)

            # Compute composite signal
            signal = compute_composite(daily_df, regime, signal_threshold=SIGNAL_THRESHOLD)

            if signal.direction == "HOLD":
                continue

            # Signal is actionable — run through risk gate
            current_price = daily_df["close"].iloc[-1]
            atr = daily_df["atr"].iloc[-1] if "atr" in daily_df.columns else current_price * 0.02

            risk = evaluate_risk(
                signal=signal,
                symbol=symbol,
                sector=sector,
                current_price=current_price,
                atr=atr,
                product="MIS",  # Intraday
            )

            if not risk.approved:
                log.debug(f"{symbol}: Signal {signal.signal:+.3f} rejected — {risk.reason}")
                continue

            signals_fired += 1

            # Execute based on mode
            order = execute_signal(
                symbol=symbol,
                exchange=exchange,
                signal=signal,
                risk=risk,
                product="MIS",
            )

            # Send Telegram notification for approval mode
            if cfg.TRADING_MODE == "approval" and order.get("status") == "pending_approval":
                send_signal(order)

        except Exception as e:
            log.warning(f"Intraday scan failed for {symbol}: {e}")

    log.info(f"─── Intraday cycle complete: {signals_fired} signals fired ───")


if __name__ == "__main__":
    run()
