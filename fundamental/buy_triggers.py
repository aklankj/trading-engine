"""
Fundamental buy trigger detection.

Checks daily if any quality stock (score ≥ 80) has dropped
significantly from its 52-week high during a Bear/HighVol regime.
"""

from core.data import fetch_candles, fetch_ltp
from core.regime import detect_regime, RegimeType
from fundamental.screener import score_watchlist, QualityScore
from execution.telegram_bot import send_fundamental_alert
from utils.logger import log
from config.settings import cfg


def check_buy_triggers(
    min_score: float = 80,
    min_drop_pct: float = 15.0,
    regime_filter: list[str] = None,
) -> list[dict]:
    """
    Scan fundamental watchlist for buy opportunities.

    A buy trigger fires when ALL conditions are met:
    1. Quality score >= min_score
    2. Price dropped >= min_drop_pct from 52-week high
    3. Current regime is in regime_filter (default: Bear, HighVol, Recovery)
    4. No red flags

    Returns list of triggered opportunities.
    """
    if regime_filter is None:
        regime_filter = ["Bear", "HighVol", "Recovery"]

    scored = score_watchlist()
    eligible = [s for s in scored if s.score >= min_score and s.signal != "avoid"]

    if not eligible:
        log.info("No eligible stocks for fundamental buy trigger")
        return []

    # Fetch live prices
    symbols = [f"{s.exchange}:{s.symbol}" for s in eligible]
    prices = fetch_ltp(symbols)

    triggers = []
    for stock in eligible:
        key = f"{stock.exchange}:{stock.symbol}"
        current_price = prices.get(key, 0)
        if current_price <= 0:
            continue

        # Get 52-week high from historical data
        try:
            # Find instrument token from watchlist
            token = _find_token(stock.symbol)
            if not token:
                continue

            candles = fetch_candles(token, interval="day", days=260)
            if candles.empty:
                continue

            high_52w = candles["high"].max()
            drop_pct = ((high_52w - current_price) / high_52w) * 100

            # Check regime on this stock's data
            regime_state = detect_regime(candles)

            stock.current_price = current_price
            stock.high_52w = high_52w
            stock.drop_from_high_pct = drop_pct

            if drop_pct >= min_drop_pct and regime_state.regime in regime_filter:
                trigger = {
                    "name": stock.name,
                    "symbol": stock.symbol,
                    "sector": stock.sector,
                    "score": stock.score,
                    "roce": stock.roce,
                    "price": current_price,
                    "high_52w": high_52w,
                    "drop_pct": round(drop_pct, 1),
                    "regime": regime_state.regime,
                    "note": stock.note,
                }
                triggers.append(trigger)
                send_fundamental_alert(trigger)
                log.info(
                    f"🏛️ FUNDAMENTAL BUY TRIGGER: {stock.name} | "
                    f"Score: {stock.score} | Drop: {drop_pct:.1f}% | "
                    f"Regime: {regime_state.regime}"
                )

        except Exception as e:
            log.warning(f"Failed to check {stock.symbol}: {e}")

    log.info(f"Fundamental scan: {len(eligible)} eligible, {len(triggers)} triggered")
    return triggers


def _find_token(symbol: str) -> int | None:
    """Find instrument token for a symbol from watchlist."""
    for item in cfg.WATCHLIST:
        if item["symbol"] == symbol:
            return item["token"]
    return None
