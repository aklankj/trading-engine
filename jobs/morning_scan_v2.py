"""
Morning scan V2 — uses tiered watchlist.

Tier 1 (NIFTY 50 + MCX + Currency): Full regime + strategy scan
Tier 2 (Next 50 + Midcaps): Regime scan only, promote strong signals
Sector indices: Rotation analysis for macro overlay

Runs at 8:45 AM IST.
"""

from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles, compute_indicators
from core.regime import detect_regime
from core.watchlist import WatchlistManager
from execution.telegram_bot import send_message, send_regime_alert
from utils.logger import log
from utils import load_json, save_json, today_str


def run():
    """Execute tiered morning scan."""
    log.info("═══ MORNING SCAN V2 STARTED ═══")

    # 1. Authenticate
    try:
        kite = get_kite()
        if kite is None:
            log.error("Authentication failed")
            send_message("⚠️ Morning scan failed: Kite login required")
            return
        profile = kite.profile()
        log.info(f"Authenticated as {profile.get('user_name', 'N/A')}")
    except Exception as e:
        log.error(f"Auth failed: {e}")
        send_message(f"⚠️ Morning scan failed: {str(e)[:100]}")
        return

    # 2. Build/refresh watchlists
    mgr = WatchlistManager()
    try:
        mgr.refresh_instruments(kite)
        watchlists = mgr.build_all()
        tier1 = watchlists["tier1"]
        tier2 = watchlists["tier2"]
        log.info(f"Watchlists: Tier 1={len(tier1)}, Tier 2={len(tier2)}, Tier 3={watchlists['tier3_count']}")
    except Exception as e:
        log.warning(f"Watchlist build failed, using defaults: {e}")
        tier1 = cfg.WATCHLIST
        tier2 = []

    # 3. Load previous regimes
    regime_state_path = cfg.DATA_DIR / "regime_state.json"
    prev_regimes = load_json(regime_state_path, default={})

    # 4. Scan Tier 1 (full scan)
    tier1_results = []
    regime_changes = []

    for item in tier1:
        symbol = item["symbol"]
        token = item.get("token", 0)
        if not token:
            continue

        try:
            df = fetch_candles(token, interval="day", days=120)
            if df.empty or len(df) < 60:
                continue

            df = compute_indicators(df)
            regime = detect_regime(df)

            tier1_results.append({
                "symbol": symbol,
                "sector": item.get("sector", "Unknown"),
                "type": item.get("type", "equity"),
                "regime": regime.regime,
                "confidence": regime.confidence,
                "ann_return": regime.ann_return,
                "ann_vol": regime.ann_volatility,
                "momentum_20d": regime.momentum_20d,
                "smoothed": regime.smoothed_signal,
                "tier": 1,
            })

            prev = prev_regimes.get(symbol, {}).get("regime", "Unknown")
            if prev != regime.regime and prev != "Unknown":
                regime_changes.append((symbol, prev, regime.regime, regime))
                # Only send Telegram for high-confidence regime changes
                if regime.confidence >= 0.65:
                    send_regime_alert(symbol, prev, regime.regime, regime)
                else:
                    log.debug(f"Low-confidence regime change for {symbol}: {prev} → {regime.regime} (conf={regime.confidence:.0%})")

            prev_regimes[symbol] = {
                "regime": regime.regime,
                "confidence": regime.confidence,
                "date": today_str(),
            }

        except Exception as e:
            log.warning(f"Tier 1 scan failed for {symbol}: {e}")

    # 5. Scan Tier 2 (regime only, lighter scan)
    tier2_results = []
    tier2_strong = []  # Strong signals to highlight

    for item in tier2:
        symbol = item["symbol"]
        token = item.get("token", 0)
        if not token or item.get("type") == "index":
            continue

        try:
            df = fetch_candles(token, interval="day", days=120)
            if df.empty or len(df) < 60:
                continue

            regime = detect_regime(df)

            result = {
                "symbol": symbol,
                "sector": item.get("sector", "Unknown"),
                "regime": regime.regime,
                "momentum_20d": regime.momentum_20d,
                "smoothed": regime.smoothed_signal,
                "tier": 2,
            }
            tier2_results.append(result)

            # Flag strong signals for Telegram
            if abs(regime.momentum_20d) > 0.15 or abs(regime.smoothed_signal) > 0.7:
                tier2_strong.append(result)

            prev_regimes[symbol] = {
                "regime": regime.regime,
                "confidence": regime.confidence,
                "date": today_str(),
            }

        except Exception as e:
            log.warning(f"Tier 2 scan failed for {symbol}: {e}")

    save_json(regime_state_path, prev_regimes)

    # 6. Build and send summary
    all_results = tier1_results + tier2_results
    _send_morning_summary(tier1_results, tier2_results, tier2_strong, regime_changes)

    log.info(
        f"═══ MORNING SCAN COMPLETE: "
        f"Tier 1: {len(tier1_results)}, Tier 2: {len(tier2_results)}, "
        f"Regime changes: {len(regime_changes)} ═══"
    )


def _send_morning_summary(
    tier1: list, tier2: list, tier2_strong: list, changes: list
):
    """Format and send the tiered morning Telegram summary."""
    regime_emoji = {
        "Bull": "🐂", "Bear": "🐻", "Sideways": "➡️",
        "HighVol": "🌊", "Recovery": "🌱", "Unknown": "❓",
    }

    # Tier 1 regime counts
    t1_regimes = {}
    for r in tier1:
        reg = r["regime"]
        t1_regimes[reg] = t1_regimes.get(reg, 0) + 1

    # Sector analysis from Tier 1
    sector_momentum = {}
    for r in tier1:
        if r["type"] == "equity":
            s = r["sector"]
            if s not in sector_momentum:
                sector_momentum[s] = []
            sector_momentum[s].append(r["momentum_20d"])

    text = f"🌅 <b>MORNING SCAN — {today_str()}</b>\n\n"

    # Tier 1 summary
    text += f"<b>📊 Tier 1 — Active Universe ({len(tier1)} instruments)</b>\n"
    for reg, count in sorted(t1_regimes.items(), key=lambda x: -x[1]):
        text += f"  {regime_emoji.get(reg, '?')} {reg}: {count}\n"

    # Regime changes
    if changes:
        text += f"\n🔄 <b>Regime changes ({len(changes)}):</b>\n"
        for sym, old, new, _ in changes[:5]:
            text += f"  {sym}: {old} → <b>{new}</b>\n"

    # Top movers Tier 1
    sorted_t1 = sorted(tier1, key=lambda x: x.get("momentum_20d", 0))
    text += "\n📉 <b>Biggest drops (Tier 1):</b>\n"
    for r in sorted_t1[:5]:
        text += f"  {r['symbol']}: {r['momentum_20d']:.1%} | {regime_emoji.get(r['regime'], '')} {r['regime']}\n"

    top_gainers = sorted(tier1, key=lambda x: x.get("momentum_20d", 0), reverse=True)
    if top_gainers and top_gainers[0].get("momentum_20d", 0) > 0.02:
        text += "\n📈 <b>Strongest (Tier 1):</b>\n"
        for r in top_gainers[:3]:
            if r.get("momentum_20d", 0) > 0.01:
                text += f"  {r['symbol']}: +{r['momentum_20d']:.1%} | {regime_emoji.get(r['regime'], '')} {r['regime']}\n"

    # Sector rotation
    if sector_momentum:
        text += "\n🏭 <b>Sector momentum (20d avg):</b>\n"
        sector_avg = {s: sum(v)/len(v) for s, v in sector_momentum.items() if len(v) >= 2}
        for s, avg in sorted(sector_avg.items(), key=lambda x: x[1])[:5]:
            text += f"  {s}: {avg:.1%}\n"

    # Tier 2 highlights
    if tier2_strong:
        text += f"\n🔍 <b>Tier 2 — Notable signals ({len(tier2)} scanned):</b>\n"
        for r in tier2_strong[:5]:
            direction = "↑" if r["momentum_20d"] > 0 else "↓"
            text += f"  {r['symbol']}: {direction} {r['momentum_20d']:.1%} | {r['regime']}\n"
    elif tier2:
        text += f"\n🔍 Tier 2: {len(tier2)} stocks scanned, no strong signals\n"

    # Commodities
    commodities = [r for r in tier1 if r.get("type") == "commodity"]
    if commodities:
        text += "\n🛢️ <b>Commodities:</b>\n"
        for r in commodities:
            text += f"  {r['symbol']}: {regime_emoji.get(r['regime'], '')} {r['regime']} | {r['momentum_20d']:.1%}\n"

    send_message(text)


if __name__ == "__main__":
    run()
