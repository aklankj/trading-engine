"""
Morning scan — runs at 8:45 AM IST before market opens.

1. Authenticates with Kite (refreshes token if needed)
2. Pulls last 60 days of data for watchlist
3. Detects regime for each instrument
4. Sends Telegram summary
"""

from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles, compute_indicators
from core.regime import detect_regime
from execution.telegram_bot import send_message, send_regime_alert
from utils.logger import log
from utils import load_json, save_json, today_str


def run():
    """Execute morning pre-market scan."""
    log.info("═══ MORNING SCAN STARTED ═══")

    # 1. Authenticate
    try:
        kite = get_kite()
        profile = kite.profile()
        log.info(f"Authenticated as {profile.get('user_name', 'N/A')}")
    except SystemExit:
        log.error("Authentication failed — need manual login")
        send_message("⚠️ <b>Morning scan failed</b>: Kite login required. Run auth manually.")
        return

    # 2. Load previous regime state for comparison
    regime_state_path = cfg.DATA_DIR / "regime_state.json"
    prev_regimes = load_json(regime_state_path, default={})

    # 3. Scan each instrument
    results = []
    regime_changes = []

    for item in cfg.WATCHLIST:
        symbol = item["symbol"]
        token = item["token"]

        try:
            df = fetch_candles(token, interval="day", days=120)
            if df.empty or len(df) < 60:
                log.warning(f"Insufficient data for {symbol}")
                continue

            df = compute_indicators(df)
            regime = detect_regime(df)

            results.append({
                "symbol": symbol,
                "regime": regime.regime,
                "confidence": regime.confidence,
                "ann_return": regime.ann_return,
                "ann_vol": regime.ann_volatility,
                "momentum_20d": regime.momentum_20d,
                "smoothed": regime.smoothed_signal,
            })

            # Check for regime change
            prev = prev_regimes.get(symbol, {}).get("regime", "Unknown")
            if prev != regime.regime and prev != "Unknown":
                regime_changes.append((symbol, prev, regime.regime, regime))
                send_regime_alert(symbol, prev, regime.regime, regime)

            # Update stored regime
            prev_regimes[symbol] = {
                "regime": regime.regime,
                "confidence": regime.confidence,
                "date": today_str(),
            }

        except Exception as e:
            log.warning(f"Failed to scan {symbol}: {e}")

    save_json(regime_state_path, prev_regimes)

    # 4. Build and send summary
    if results:
        _send_morning_summary(results, regime_changes)

    log.info(f"═══ MORNING SCAN COMPLETE: {len(results)} instruments scanned ═══")


def _send_morning_summary(results: list[dict], changes: list):
    """Format and send the morning Telegram summary."""
    regime_emoji = {
        "Bull": "🐂", "Bear": "🐻", "Sideways": "➡️",
        "HighVol": "🌊", "Recovery": "🌱", "Unknown": "❓",
    }

    # Count regimes
    regime_counts = {}
    for r in results:
        reg = r["regime"]
        regime_counts[reg] = regime_counts.get(reg, 0) + 1

    text = f"🌅 <b>MORNING SCAN — {today_str()}</b>\n\n"

    # Regime distribution
    text += "<b>Market regimes:</b>\n"
    for reg, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        text += f"  {regime_emoji.get(reg, '?')} {reg}: {count} instruments\n"

    # Regime changes
    if changes:
        text += f"\n🔄 <b>Regime changes ({len(changes)}):</b>\n"
        for sym, old, new, _ in changes:
            text += f"  {sym}: {old} → <b>{new}</b>\n"

    # Top movers by momentum
    sorted_by_mom = sorted(results, key=lambda x: abs(x["momentum_20d"]), reverse=True)
    text += "\n📊 <b>Strongest signals:</b>\n"
    for r in sorted_by_mom[:5]:
        direction = "↑" if r["momentum_20d"] > 0 else "↓"
        text += (f"  {r['symbol']}: {direction} {r['momentum_20d']:.1%} (20d) | "
                 f"{regime_emoji.get(r['regime'], '')} {r['regime']}\n")

    send_message(text)


if __name__ == "__main__":
    run()
