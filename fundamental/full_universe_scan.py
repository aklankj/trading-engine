"""
Full universe fundamental scan + news sentiment.

Pass 1: Filter 9,336 NSE instruments to tradeable equities
Pass 2: Scrape Screener.in for each (with resume support)
Pass 3: Google News + OpenRouter sentiment for top scorers

Designed to run as a background job (~30-45 min for ~1000 stocks).
Saves progress so it can resume if interrupted.
"""

import re
import time
import json
import requests
import feedparser
from pathlib import Path
from datetime import datetime

from config.settings import cfg
from fundamental.screener_scraper import scrape_company, score_company
from utils.logger import log
from utils import save_json, load_json

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
PROGRESS_FILE = cfg.DATA_DIR / "fundscan_progress.json"
RESULTS_FILE = cfg.DATA_DIR / "fundamental_full_universe.json"


def get_all_equity_symbols() -> list[str]:
    """
    Filter instrument cache to tradeable equities only.
    Removes ETFs, bonds, warrants, rights, debentures, etc.
    """
    cache = load_json(cfg.DATA_DIR / "instrument_cache.json", default={})
    nse = cache.get("nse", [])

    # Patterns to exclude
    exclude_patterns = [
        "ETF", "LIQUID", "GILT", "NIFTY", "BANK", "GOLD",
        "SILVER", "-RE", "-PP", "-W1", "-W2", "-W3",
        "TBILL", "GSEC", "SDL", "BOND",
        "SGBMAR", "SGBAPR", "SGBJUN", "SGBJUL", "SGBAUG", "SGBSEP",
        "SGBOCT", "SGBNOV", "SGBDEC", "SGBJAN", "SGBFEB",
        "IRGOLD", "IRSILV",
    ]

    symbols = []
    for inst in nse:
        sym = inst.get("symbol", "")
        if not sym:
            continue
        # Skip if matches any exclude pattern
        upper = sym.upper()
        if any(pat in upper for pat in exclude_patterns):
            continue
        # Skip very short symbols (likely indices)
        if len(sym) < 2:
            continue
        # Skip if name contains bond/debenture indicators
        name = inst.get("name", "").upper()
        if any(x in name for x in ["DEBENTURE", "BOND", "RIGHTS", "WARRANT"]):
            continue
        symbols.append(sym)

    log.info(f"Filtered {len(nse)} NSE instruments to {len(symbols)} likely equities")
    return symbols


def load_progress() -> dict:
    """Load scan progress for resume support."""
    return load_json(PROGRESS_FILE, default={
        "completed": [],
        "results": [],
        "started_at": None,
        "last_updated": None,
    })


def save_progress(progress: dict):
    """Save scan progress."""
    progress["last_updated"] = datetime.now().isoformat()
    save_json(PROGRESS_FILE, progress)


def fetch_news(symbol: str, company_name: str = "") -> list[dict]:
    """
    Fetch recent news for a company via Google News RSS.
    Free, no API key needed.
    """
    query = f"{symbol} stock" if not company_name else f"{company_name} stock"
    url = f"https://news.google.com/rss/search?q={query}+India&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:5]:
            news.append({
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
            })
        return news
    except Exception as e:
        log.debug(f"News fetch failed for {symbol}: {e}")
        return []


def analyze_news_sentiment(symbol: str, news: list[dict], company_data: dict) -> dict:
    """
    Use OpenRouter to analyze news sentiment and classify as
    temporary vs structural impact.
    """
    if not news or not cfg.OPENROUTER_API_KEY:
        return {"sentiment": 0, "is_temporary": True, "analysis": "No news or API key"}

    headlines = "\n".join([f"- {n['title']} ({n['source']})" for n in news[:5]])
    roce = company_data.get("ROCE", 0)
    score = company_data.get("quality_score", 0)

    prompt = f"""Analyze these recent news headlines for {symbol} (Indian stock).
Company fundamentals: ROCE={roce}%, Quality Score={score}/100

Headlines:
{headlines}

Respond in this exact JSON format only, no other text:
{{
  "sentiment": 0.5,
  "is_temporary": true,
  "summary": "one line summary",
  "impact_on_stock": "positive/negative/neutral",
  "buy_opportunity": true
}}

sentiment: -1.0 (very negative) to +1.0 (very positive)
is_temporary: true if bad news is temporary (one-time event, fine, single quarter miss), false if structural (market share loss, governance issue, tech disruption)
buy_opportunity: true if stock is high quality AND bad news is temporary (buying opportunity)"""

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            },
            timeout=20,
        )

        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            clean = re.sub(r"```json\s*|```\s*", "", text).strip()
            return json.loads(clean)
    except Exception as e:
        log.debug(f"News analysis failed for {symbol}: {e}")

    return {"sentiment": 0, "is_temporary": True, "analysis": "Analysis failed"}


def run_full_universe_scan(max_stocks: int = None, skip_news: bool = False):
    """
    Full universe scan with resume support.

    Args:
        max_stocks: Limit number of stocks (None = all)
        skip_news: Skip news analysis (faster)
    """
    log.info("═══ FULL UNIVERSE FUNDAMENTAL SCAN ═══")

    # Get all equity symbols
    all_symbols = get_all_equity_symbols()
    if max_stocks:
        all_symbols = all_symbols[:max_stocks]

    # Load progress for resume
    progress = load_progress()
    completed = set(progress.get("completed", []))
    results = progress.get("results", [])

    if not progress.get("started_at"):
        progress["started_at"] = datetime.now().isoformat()

    remaining = [s for s in all_symbols if s not in completed]
    log.info(f"Total: {len(all_symbols)} | Already done: {len(completed)} | Remaining: {len(remaining)}")

    # Telegram start notification
    try:
        from execution.telegram_bot import send_message
        send_message(
            f"🏛️ <b>Full Universe Scan Started</b>\n"
            f"Total stocks: <code>{len(all_symbols)}</code>\n"
            f"Already completed: <code>{len(completed)}</code>\n"
            f"Remaining: <code>{len(remaining)}</code>\n"
            f"Estimated time: <code>{len(remaining) * 1.5 / 60:.0f} min</code>"
        )
    except Exception:
        pass

    errors = 0
    batch_count = 0

    for i, sym in enumerate(remaining):
        try:
            data = scrape_company(sym)

            if "error" in data:
                errors += 1
                completed.add(sym)
                if errors % 50 == 0:
                    log.info(f"  {errors} errors so far (likely non-equity symbols)")
                continue

            data = score_company(data)

            # Only keep companies with meaningful data
            if data.get("Market Cap", 0) > 0:
                results.append(data)

            completed.add(sym)
            batch_count += 1

            # Log progress every 50 stocks
            if batch_count % 50 == 0:
                log.info(f"  Progress: {len(completed)}/{len(all_symbols)} | Valid: {len(results)} | Errors: {errors}")
                save_progress({
                    "completed": list(completed),
                    "results": results,
                    "started_at": progress["started_at"],
                })

            time.sleep(1.2)

        except Exception as e:
            errors += 1
            completed.add(sym)
            log.debug(f"  {sym}: {e}")

    # Sort by score
    results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # News analysis for top 50 scorers
    if not skip_news:
        log.info(f"Running news analysis for top 50 companies...")
        top_50 = [r for r in results if r.get("quality_score", 0) >= 60][:50]

        for i, company in enumerate(top_50):
            sym = company.get("symbol", "")
            name = company.get("company_name", sym)

            log.info(f"  News [{i+1}/{len(top_50)}]: {sym}")
            news = fetch_news(sym, name)
            company["recent_news"] = news

            if news and cfg.OPENROUTER_API_KEY:
                sentiment = analyze_news_sentiment(sym, news, company)
                company["news_sentiment"] = sentiment

                # Adjust score based on news
                sent_val = sentiment.get("sentiment", 0)
                is_temp = sentiment.get("is_temporary", True)
                buy_opp = sentiment.get("buy_opportunity", False)

                if buy_opp and company.get("quality_score", 0) >= 70:
                    company["quality_score"] = min(100, company["quality_score"] + 5)
                    company["news_flag"] = "BUY_OPPORTUNITY"
                elif not is_temp and sent_val < -0.3:
                    company["quality_score"] = max(0, company["quality_score"] - 10)
                    company["news_flag"] = "STRUCTURAL_RISK"

                time.sleep(1)

        # Re-sort after news adjustments
        results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Save final results
    save_json(RESULTS_FILE, {
        "scan_date": datetime.now().isoformat(),
        "total_symbols": len(all_symbols),
        "successful_scrapes": len(results),
        "errors": errors,
        "results": results,
    })

    # Build watchlist from high scorers
    watchlist = []
    for c in results:
        if c.get("quality_score", 0) >= 65:
            entry = {
                "name": c.get("company_name", c.get("symbol", "")),
                "symbol": c.get("symbol", ""),
                "exchange": "NSE",
                "sector": c.get("sector", c.get("industry", "Unknown")),
                "roce": c.get("ROCE", 0),
                "roe": c.get("ROE", 0),
                "pe": c.get("Stock P/E", 0),
                "market_cap": c.get("Market Cap", 0),
                "quality_score": c.get("quality_score", 0),
                "signal": c.get("signal", "watch"),
                "news_flag": c.get("news_flag", "none"),
            }
            if c.get("news_sentiment"):
                entry["news_summary"] = c["news_sentiment"].get("summary", "")
            watchlist.append(entry)

    save_json(cfg.DATA_DIR / "fundamental_watchlist.json", watchlist)

    # Clear progress file (scan complete)
    save_json(PROGRESS_FILE, {"completed": [], "results": [], "status": "complete",
                               "last_complete_scan": datetime.now().isoformat()})

    # Stats
    score_90 = len([r for r in results if r.get("quality_score", 0) >= 90])
    score_80 = len([r for r in results if r.get("quality_score", 0) >= 80])
    score_70 = len([r for r in results if r.get("quality_score", 0) >= 70])
    buy_opps = len([r for r in results if r.get("news_flag") == "BUY_OPPORTUNITY"])
    struct_risks = len([r for r in results if r.get("news_flag") == "STRUCTURAL_RISK"])

    log.info(
        f"═══ FULL UNIVERSE SCAN COMPLETE ═══\n"
        f"  Scanned: {len(all_symbols)} symbols\n"
        f"  Valid companies: {len(results)}\n"
        f"  Score >= 90: {score_90}\n"
        f"  Score >= 80: {score_80}\n"
        f"  Score >= 70: {score_70}\n"
        f"  Watchlist: {len(watchlist)}\n"
        f"  News buy opportunities: {buy_opps}\n"
        f"  News structural risks: {struct_risks}"
    )

    # Telegram summary
    try:
        from execution.telegram_bot import send_message
        msg = (
            f"🏛️ <b>Full Universe Scan Complete</b>\n\n"
            f"Scanned: <code>{len(all_symbols)}</code> symbols\n"
            f"Valid companies: <code>{len(results)}</code>\n"
            f"Score ≥ 80: <code>{score_80}</code>\n"
            f"Score ≥ 70: <code>{score_70}</code>\n"
            f"Watchlist: <code>{len(watchlist)}</code>\n\n"
        )
        if buy_opps:
            msg += f"🟢 Buy opportunities (quality + temp bad news): <code>{buy_opps}</code>\n"
        if struct_risks:
            msg += f"🔴 Structural risks flagged: <code>{struct_risks}</code>\n"

        msg += "\n<b>Top 10 quality companies:</b>\n"
        for c in results[:10]:
            flag = ""
            if c.get("news_flag") == "BUY_OPPORTUNITY": flag = " 🟢"
            elif c.get("news_flag") == "STRUCTURAL_RISK": flag = " 🔴"
            msg += (
                f"  <b>{c.get('symbol', '?')}</b>: "
                f"Score={c.get('quality_score', 0):.0f} "
                f"ROCE={c.get('ROCE', 0):.1f}% "
                f"PE={c.get('Stock P/E', 0):.1f}{flag}\n"
            )

        # Show buy opportunities specifically
        buy_ops = [r for r in results if r.get("news_flag") == "BUY_OPPORTUNITY"]
        if buy_ops:
            msg += "\n<b>🟢 Buy opportunities (quality + temporary dip):</b>\n"
            for c in buy_ops[:5]:
                ns = c.get("news_sentiment", {})
                msg += (
                    f"  <b>{c.get('symbol', '?')}</b> (Score={c.get('quality_score', 0):.0f}): "
                    f"{ns.get('summary', 'N/A')}\n"
                )

        send_message(msg)
    except Exception as e:
        log.warning(f"Telegram failed: {e}")

    return results


if __name__ == "__main__":
    import sys
    max_n = None
    skip_news = "--no-news" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            max_n = int(arg)
    run_full_universe_scan(max_stocks=max_n, skip_news=skip_news)
