"""
Screener.in company page scraper.
Scrapes individual company pages (no login needed).
Extracts ROCE, ROE, PE, growth, debt, and scores each company.
"""

import re
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

from config.settings import cfg
from utils.logger import log
from utils import save_json, load_json

SCREENER_BASE = "https://www.screener.in"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def scrape_company(symbol: str, consolidated: bool = True) -> dict:
    """Scrape all fundamental data for one company."""
    suffix = "/consolidated/" if consolidated else "/"
    url = f"{SCREENER_BASE}/company/{symbol}{suffix}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404 and consolidated:
            resp = requests.get(f"{SCREENER_BASE}/company/{symbol}/", headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {"symbol": symbol, "error": f"HTTP {resp.status_code}"}

        soup = BeautifulSoup(resp.text, "html.parser")
        data = {"symbol": symbol, "scraped_at": datetime.now().isoformat()}

        # Company name
        h1 = soup.find("h1")
        if h1:
            data["company_name"] = h1.get_text(strip=True)

        # Top ratios
        ratios = soup.find("ul", id="top-ratios")
        if ratios:
            for li in ratios.find_all("li"):
                name = li.find("span", class_="name")
                val = li.find("span", class_="number")
                if name and val:
                    data[name.text.strip()] = _parse_num(val.text.strip())

        # Growth tables (Sales, Profit, Price CAGR, ROE)
        sections = soup.find_all("div", class_="ranges-table")
        growth_labels = ["sales_growth", "profit_growth", "price_cagr", "roe_trend"]
        for i, section in enumerate(sections):
            label = growth_labels[i] if i < len(growth_labels) else f"section_{i}"
            rows = section.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    period = cells[0].text.strip().replace(":", "")
                    value = _parse_num(cells[-1].text.strip())
                    data[f"{label}_{period}"] = value

        # Pros count
        data["pros_count"] = len(soup.select(".pros li"))
        data["cons_count"] = len(soup.select(".cons li"))

        # Cons text for analysis
        cons = [li.text.strip() for li in soup.select(".cons li")]
        data["cons_text"] = cons

        # Sector from about section
        about = soup.find("div", class_="about")
        if about:
            for link in about.find_all("a"):
                href = link.get("href", "")
                if "/sector/" in href:
                    data["sector"] = link.text.strip()
                elif "/industry/" in href:
                    data["industry"] = link.text.strip()

        return data

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def _parse_num(value: str) -> float:
    if not value:
        return 0.0
    try:
        cleaned = value.replace(",", "").replace("%", "").replace("Cr.", "").replace("Cr", "").replace("₹", "").strip()
        parts = cleaned.split("/")
        if len(parts) > 1:
            return float(parts[0].strip())
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def score_company(data: dict) -> dict:
    """Score company 0-100 on quality metrics."""
    score = 0.0

    roce = data.get("ROCE", 0)
    if roce > 25: score += 25
    elif roce > 15: score += 17.5
    elif roce > 10: score += 10
    else: score += 5

    roe = data.get("ROE", 0)
    if roe > 20: score += 15
    elif roe > 12: score += 10.5
    else: score += 4.5

    de = data.get("Debt to equity", data.get("Debt / Equity", 999))
    if de < 0.1: score += 10
    elif de < 0.5: score += 7
    elif de < 1.0: score += 4
    else: score += 1

    sg = data.get("sales_growth_5 Years", data.get("sales_growth_5Years", 0))
    if sg > 18: score += 10
    elif sg > 10: score += 7
    else: score += 3

    pg = data.get("profit_growth_5 Years", data.get("profit_growth_5Years", 0))
    if pg > 18: score += 10
    elif pg > 10: score += 7
    else: score += 3

    pe = data.get("Stock P/E", 0)
    if 0 < pe < 15: score += 10
    elif 0 < pe < 25: score += 7
    elif 0 < pe < 40: score += 4
    else: score += 1

    dy = data.get("Dividend Yield", 0)
    if dy > 2: score += 5
    elif dy > 1: score += 3.5
    else: score += 1

    mcap = data.get("Market Cap", 0)
    if mcap > 50000: score += 5
    elif mcap > 10000: score += 3.5
    elif mcap > 2000: score += 2
    else: score += 0.5

    if data.get("cons_count", 5) == 0: score += 5
    elif data.get("cons_count", 5) <= 1: score += 3
    if data.get("pros_count", 0) >= 3: score += 5
    elif data.get("pros_count", 0) >= 1: score += 2

    data["quality_score"] = round(min(100, score), 1)

    if score >= 85: data["signal"] = "buy-dip"
    elif score >= 70: data["signal"] = "hold"
    elif score >= 55: data["signal"] = "watch"
    else: data["signal"] = "avoid"

    return data


def run_full_fundamental_scan(symbols: list[str] = None) -> list[dict]:
    """
    Scan companies by scraping individual Screener.in pages.
    If no symbols given, uses Tier 1 + Tier 2 watchlist.
    """
    log.info("═══ FUNDAMENTAL SCAN STARTED ═══")

    if not symbols:
        wl_data = load_json(cfg.DATA_DIR / "tiered_watchlist.json", default={})
        tier1 = [w["symbol"] for w in wl_data.get("tier1", []) if w.get("type") == "equity"]
        tier2 = [w["symbol"] for w in wl_data.get("tier2", []) if w.get("type", "equity") == "equity"]
        symbols = list(set(tier1 + tier2))
        if not symbols:
            symbols = ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","BAJFINANCE",
                       "ASIANPAINT","TITAN","PIDILITIND","SBIN","ITC","MARUTI",
                       "SUNPHARMA","LT","WIPRO","BHARTIARTL","KOTAKBANK","AXISBANK",
                       "HCLTECH","HINDUNILVR","NESTLEIND","CIPLA","DRREDDY","DIVISLAB",
                       "BRITANNIA","COALINDIA","NTPC","POWERGRID","ONGC","BPCL",
                       "TATASTEEL","JSWSTEEL","HINDALCO","ULTRACEMCO","GRASIM",
                       "ADANIPORTS","CDSL","DMART","ASTRAL","POLYCAB","HAVELLS"]

    log.info(f"Scanning {len(symbols)} companies on Screener.in...")
    results = []
    errors = 0

    for i, sym in enumerate(symbols):
        try:
            log.info(f"  [{i+1}/{len(symbols)}] {sym}")
            data = scrape_company(sym)

            if "error" in data:
                errors += 1
                log.warning(f"  {sym}: {data['error']}")
                continue

            data = score_company(data)
            results.append(data)
            time.sleep(1.2)

        except Exception as e:
            errors += 1
            log.warning(f"  {sym} failed: {e}")

    results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Save full results
    save_json(cfg.DATA_DIR / "fundamental_scan.json", {
        "scan_date": datetime.now().isoformat(),
        "total_scanned": len(symbols),
        "successful": len(results),
        "errors": errors,
        "results": results,
    })

    # Build fundamental watchlist from high scorers
    watchlist = []
    for c in results:
        if c.get("quality_score", 0) >= 70:
            watchlist.append({
                "name": c.get("company_name", c.get("symbol", "")),
                "symbol": c.get("symbol", ""),
                "exchange": "NSE",
                "sector": c.get("sector", c.get("industry", "Unknown")),
                "roce": c.get("ROCE", 0),
                "roe": c.get("ROE", 0),
                "pe": c.get("Stock P/E", 0),
                "market_cap": c.get("Market Cap", 0),
                "debt_equity": c.get("Debt to equity", 0),
                "dividend_yield": c.get("Dividend Yield", 0),
                "quality_score": c.get("quality_score", 0),
                "signal": c.get("signal", "watch"),
                "sales_growth_5y": c.get("sales_growth_5 Years", 0),
                "profit_growth_5y": c.get("profit_growth_5 Years", 0),
                "pros_count": c.get("pros_count", 0),
                "cons_count": c.get("cons_count", 0),
            })
    save_json(cfg.DATA_DIR / "fundamental_watchlist.json", watchlist)

    # Summary stats
    score_80 = len([r for r in results if r.get("quality_score", 0) >= 80])
    score_70 = len([r for r in results if r.get("quality_score", 0) >= 70])
    log.info(
        f"═══ FUNDAMENTAL SCAN COMPLETE ═══\n"
        f"  Scanned: {len(results)}/{len(symbols)} companies\n"
        f"  Score >= 80: {score_80}\n"
        f"  Score >= 70: {score_70}\n"
        f"  Watchlist: {len(watchlist)} companies"
    )

    # Telegram summary
    try:
        from execution.telegram_bot import send_message
        msg = (
            f"🏛️ <b>Fundamental Scan Complete</b>\n\n"
            f"Screened: <code>{len(results)}</code> companies\n"
            f"Score ≥ 80: <code>{score_80}</code>\n"
            f"Score ≥ 70: <code>{score_70}</code>\n"
            f"Watchlist: <code>{len(watchlist)}</code>\n\n"
        )
        if results[:5]:
            msg += "<b>Top 5:</b>\n"
            for c in results[:5]:
                msg += (
                    f"  <b>{c.get('symbol', '?')}</b>: "
                    f"Score={c.get('quality_score', 0):.0f} "
                    f"ROCE={c.get('ROCE', 0):.1f}% "
                    f"PE={c.get('Stock P/E', 0):.1f}\n"
                )
        send_message(msg)
    except Exception:
        pass

    return results


if __name__ == "__main__":
    results = run_full_fundamental_scan()
    print(f"\nTop 10:")
    for c in results[:10]:
        print(f"  {c.get('symbol', '?'):15s} Score={c.get('quality_score', 0):5.1f} "
              f"ROCE={c.get('ROCE', 0):>6.1f}% PE={c.get('Stock P/E', 0):>6.1f}")
