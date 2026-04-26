"""
Bear Market Opportunity Scanner.

Finds high-quality companies trading at significant discounts.
The thesis: in broad Bear markets, quality gets dragged down with junk.
The quality recovers. The junk doesn't.

Combines:
  1. Fundamental quality score (ROCE, ROE, debt, growth)
  2. Price discount from 52-week high
  3. Regime context (Bear/HighVol = discount window)
  4. Historical PE band (cheap relative to itself?)
  5. Sector oversold analysis

Runs daily at 3 PM alongside buy triggers.
"""

import time
import json
from datetime import datetime
from config.settings import cfg
from core.auth import get_kite
from core.data import fetch_candles
from core.regime import detect_regime
from utils.logger import log
from utils import load_json, save_json, fmt_inr


def scan_bear_opportunities(
    min_quality: float = 55,
    min_drop_pct: float = 10,
    min_mcap: float = 1000,
) -> list[dict]:
    """
    Find quality companies trading at bear-market discounts.
    
    Args:
        min_quality: Minimum fundamental quality score
        min_drop_pct: Minimum % drop from 52-week high
        min_mcap: Minimum market cap in Crores
    """
    log.info("═══ BEAR MARKET OPPORTUNITY SCAN ═══")
    
    # Load fundamental data
    fund_data = load_json(cfg.DATA_DIR / "fundamental_full_universe.json", default={})
    fund_results = fund_data.get("results", [])
    
    if not fund_results:
        # Try the smaller scan
        fund_data = load_json(cfg.DATA_DIR / "fundamental_scan.json", default={})
        fund_results = fund_data.get("results", [])
    
    if not fund_results:
        log.warning("No fundamental data found. Run --fullscan first.")
        return []
    
    # Filter to quality companies
    quality_stocks = [
        r for r in fund_results
        if r.get("quality_score", 0) >= min_quality
        and r.get("Market Cap", 0) >= min_mcap
        and r.get("ROCE", 0) > 0
    ]
    
    log.info(f"Quality filter: {len(quality_stocks)} companies with score >= {min_quality} and mcap >= {min_mcap} Cr")
    
    if not quality_stocks:
        log.warning("No quality stocks found above threshold")
        return []
    
    # Get Kite for price data
    try:
        kite = get_kite()
        if kite is None:
            log.error("Kite auth failed")
            return []
    except Exception:
        log.error("Kite auth failed")
        return []
    
    # Load instrument cache for tokens
    cache = load_json(cfg.DATA_DIR / "instrument_cache.json", default={})
    nse_instruments = {i["symbol"]: i["token"] for i in cache.get("nse", [])}
    
    opportunities = []
    
    for i, stock in enumerate(quality_stocks):
        symbol = stock.get("symbol", "")
        if not symbol:
            continue
        
        token = nse_instruments.get(symbol, 0)
        if not token:
            continue
        
        try:
            # Fetch 1 year of daily data
            df = fetch_candles(token, interval="day", days=365)
            if df.empty or len(df) < 60:
                continue
            
            current_price = df["close"].iloc[-1]
            high_52w = df["high"].max()
            low_52w = df["low"].min()
            
            # Calculate discount from high
            drop_pct = ((high_52w - current_price) / high_52w) * 100
            
            if drop_pct < min_drop_pct:
                continue
            
            # Where in the 52-week range is it?
            range_52w = high_52w - low_52w
            position_in_range = ((current_price - low_52w) / range_52w * 100) if range_52w > 0 else 50
            
            # Detect regime
            regime = detect_regime(df)
            
            # PE analysis
            pe = stock.get("Stock P/E", 0)
            roce = stock.get("ROCE", 0)
            roe = stock.get("ROE", 0)
            score = stock.get("quality_score", 0)
            mcap = stock.get("Market Cap", 0)
            de = stock.get("Debt to equity", 0)
            
            # Earnings yield vs bond yield (10Y India ~7%)
            earnings_yield = (1 / pe * 100) if pe > 0 else 0
            ey_spread = earnings_yield - 7.0  # Spread over risk-free rate
            
            # Opportunity score: combines quality + discount + valuation
            opp_score = 0
            
            # Quality contribution (0-40 points)
            opp_score += min(40, score * 0.45)
            
            # Discount contribution (0-30 points)
            if drop_pct >= 40: opp_score += 30
            elif drop_pct >= 30: opp_score += 25
            elif drop_pct >= 20: opp_score += 20
            elif drop_pct >= 15: opp_score += 15
            else: opp_score += 10
            
            # Valuation contribution (0-20 points)
            if ey_spread > 5: opp_score += 20
            elif ey_spread > 3: opp_score += 15
            elif ey_spread > 1: opp_score += 10
            elif ey_spread > 0: opp_score += 5
            
            # Low debt bonus (0-10 points)
            if de < 0.1: opp_score += 10
            elif de < 0.3: opp_score += 7
            elif de < 0.5: opp_score += 5
            elif de < 1.0: opp_score += 2
            
            # Penalty for extreme Bear with high vol (risky to catch falling knife)
            if regime.ann_volatility > 0.4:
                opp_score -= 10
            
            # Bonus if regime is showing recovery signs
            if regime.regime == "Recovery":
                opp_score += 10
            elif regime.regime == "Sideways" and regime.smoothed_signal > 0:
                opp_score += 5
            
            # 20-day and 5-day momentum for timing
            mom_20d = (current_price / df["close"].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            mom_5d = (current_price / df["close"].iloc[-6] - 1) * 100 if len(df) > 6 else 0
            
            # Is the bleeding slowing? (good sign for entry)
            bleeding_slowing = mom_5d > mom_20d and mom_20d < 0
            if bleeding_slowing:
                opp_score += 5
            
            opportunity = {
                "symbol": symbol,
                "company_name": stock.get("company_name", symbol),
                "sector": stock.get("sector", stock.get("industry", "Unknown")),
                "quality_score": score,
                "opportunity_score": round(opp_score, 1),
                "current_price": round(current_price, 2),
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "drop_from_high_pct": round(drop_pct, 1),
                "position_in_52w_range": round(position_in_range, 1),
                "pe": round(pe, 1),
                "earnings_yield": round(earnings_yield, 1),
                "ey_spread_vs_bonds": round(ey_spread, 1),
                "roce": round(roce, 1),
                "roe": round(roe, 1),
                "debt_equity": de,
                "market_cap_cr": mcap,
                "regime": regime.regime,
                "regime_confidence": round(regime.confidence, 2),
                "momentum_20d": round(mom_20d, 1),
                "momentum_5d": round(mom_5d, 1),
                "bleeding_slowing": bleeding_slowing,
                "smoothed_signal": round(regime.smoothed_signal, 2),
            }
            opportunities.append(opportunity)
            
            # Rate limit Kite API
            time.sleep(0.4)
            
        except Exception as e:
            log.debug(f"  {symbol}: {e}")
            continue
    
    # Sort by opportunity score
    opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
    
    # Save results
    save_json(cfg.DATA_DIR / "bear_opportunities.json", {
        "scan_date": datetime.now().isoformat(),
        "market_context": f"Bear market: majority of stocks in Bear/HighVol regime",
        "total_quality_stocks": len(quality_stocks),
        "opportunities_found": len(opportunities),
        "opportunities": opportunities,
    })
    
    # Categorize
    strong_buys = [o for o in opportunities if o["opportunity_score"] >= 70]
    good_buys = [o for o in opportunities if 55 <= o["opportunity_score"] < 70]
    watchlist = [o for o in opportunities if 40 <= o["opportunity_score"] < 55]
    
    log.info(
        f"═══ BEAR OPPORTUNITY SCAN COMPLETE ═══\n"
        f"  Quality companies checked: {len(quality_stocks)}\n"
        f"  Opportunities found: {len(opportunities)}\n"
        f"  Strong buys (score >= 70): {len(strong_buys)}\n"
        f"  Good buys (55-70): {len(good_buys)}\n"
        f"  Watchlist (40-55): {len(watchlist)}"
    )
    
    # Send Telegram report
    _send_opportunity_report(opportunities, strong_buys, good_buys, watchlist)
    
    return opportunities


def _send_opportunity_report(all_opps, strong, good, watch):
    """Send formatted opportunity report to Telegram."""
    try:
        from execution.telegram_bot import send_message
        
        msg = (
            f"🏛️📉 <b>BEAR MARKET OPPORTUNITIES</b>\n"
            f"<i>Quality companies at discount prices</i>\n\n"
            f"Found: <code>{len(all_opps)}</code> opportunities\n"
            f"Strong buys: <code>{len(strong)}</code> | "
            f"Good buys: <code>{len(good)}</code> | "
            f"Watchlist: <code>{len(watch)}</code>\n"
        )
        
        if strong:
            msg += "\n🟢 <b>STRONG BUYS (highest conviction):</b>\n"
            for o in strong[:10]:
                timing = "⬆️ bleeding slowing" if o["bleeding_slowing"] else "⬇️ still falling"
                msg += (
                    f"\n  <b>{o['symbol']}</b> — {o.get('company_name', '')[:25]}\n"
                    f"  💰 ₹{o['current_price']:,.0f} (down {o['drop_from_high_pct']:.0f}% from ₹{o['high_52w']:,.0f})\n"
                    f"  📊 ROCE={o['roce']:.0f}% | PE={o['pe']:.0f} | D/E={o['debt_equity']:.1f}\n"
                    f"  🎯 Quality={o['quality_score']:.0f} | Opp={o['opportunity_score']:.0f} | {timing}\n"
                    f"  📍 {o['regime']} | EY spread: {o['ey_spread_vs_bonds']:+.1f}% vs bonds\n"
                )
        
        if good:
            msg += "\n🟡 <b>GOOD BUYS:</b>\n"
            for o in good[:10]:
                msg += (
                    f"  <b>{o['symbol']}</b>: ₹{o['current_price']:,.0f} "
                    f"(↓{o['drop_from_high_pct']:.0f}%) "
                    f"ROCE={o['roce']:.0f}% "
                    f"Opp={o['opportunity_score']:.0f}\n"
                )
        
        if watch:
            msg += "\n👀 <b>WATCHLIST (wait for better entry):</b>\n"
            for o in watch[:8]:
                msg += (
                    f"  {o['symbol']}: ↓{o['drop_from_high_pct']:.0f}% "
                    f"ROCE={o['roce']:.0f}% "
                    f"Opp={o['opportunity_score']:.0f}\n"
                )
        
        # Sector breakdown
        sectors = {}
        for o in all_opps:
            s = o.get("sector", "Unknown")
            if s not in sectors:
                sectors[s] = {"count": 0, "avg_drop": 0}
            sectors[s]["count"] += 1
            sectors[s]["avg_drop"] += o["drop_from_high_pct"]
        
        for s in sectors:
            sectors[s]["avg_drop"] /= sectors[s]["count"]
        
        sorted_sectors = sorted(sectors.items(), key=lambda x: -x[1]["avg_drop"])
        if sorted_sectors:
            msg += "\n🏭 <b>Most discounted sectors:</b>\n"
            for s, data in sorted_sectors[:5]:
                msg += f"  {s}: avg ↓{data['avg_drop']:.0f}% ({data['count']} stocks)\n"
        
        msg += (
            f"\n💡 <b>Strategy:</b> In Bear markets, accumulate quality gradually.\n"
            f"Don't go all-in at once. Buy in 3-4 tranches over weeks.\n"
            f"Focus on stocks where bleeding is slowing (⬆️ indicator)."
        )
        
        send_message(msg)
        
    except Exception as e:
        log.warning(f"Telegram report failed: {e}")


if __name__ == "__main__":
    import sys
    min_q = 55
    min_drop = 10
    for arg in sys.argv[1:]:
        if arg.startswith("--min-quality="):
            min_q = float(arg.split("=")[1])
        elif arg.startswith("--min-drop="):
            min_drop = float(arg.split("=")[1])
    scan_bear_opportunities(min_quality=min_q, min_drop_pct=min_drop)
