"""
Weekly research scan — runs Sunday evening.
Quarterly fundamental rescan — runs after results season.
"""

from research.scanner import run_research_scan
from fundamental.screener import score_watchlist, save_fundamental_watchlist
from fundamental.buy_triggers import check_buy_triggers
from execution.telegram_bot import send_message
from utils.logger import log


def run_weekly_research():
    """Sunday evening: scan for new papers and strategies."""
    log.info("═══ WEEKLY RESEARCH SCAN ═══")
    papers = run_research_scan()
    high_priority = [p for p in papers if p.get("priority_score", 0) >= 7]

    summary = (
        f"📚 <b>Weekly Research Scan Complete</b>\n\n"
        f"Papers discovered: <code>{len(papers)}</code>\n"
        f"High priority: <code>{len(high_priority)}</code>\n"
    )
    if high_priority:
        summary += "\n<b>Top finds:</b>\n"
        for p in high_priority[:3]:
            summary += f"  • {p['title'][:70]}...\n"
            ev = p.get("evaluation", {})
            if isinstance(ev, dict):
                summary += f"    Sharpe: {ev.get('reported_sharpe', 'N/A')} | Priority: {p.get('priority_score', 0)}/10\n"

    send_message(summary)
    log.info(f"Research scan done: {len(papers)} papers, {len(high_priority)} high-priority")


def run_quarterly_fundamental():
    """Results season: rescore watchlist and check for buy opportunities."""
    log.info("═══ QUARTERLY FUNDAMENTAL RESCAN ═══")

    scored = score_watchlist()

    # Send summary
    buy_candidates = [s for s in scored if s.signal == "buy-dip"]
    avoid = [s for s in scored if s.signal == "avoid"]

    summary = (
        f"🏛️ <b>Quarterly Fundamental Rescan</b>\n\n"
        f"Companies scored: <code>{len(scored)}</code>\n"
        f"Buy candidates: <code>{len(buy_candidates)}</code>\n"
        f"Avoid: <code>{len(avoid)}</code>\n\n"
    )
    if buy_candidates:
        summary += "<b>Top quality (buy on dip):</b>\n"
        for s in buy_candidates[:5]:
            summary += f"  {s.name}: Score={s.score} ROCE={s.roce}%\n"

    send_message(summary)

    # Check for active buy triggers
    triggers = check_buy_triggers()
    log.info(f"Fundamental rescan done: {len(scored)} scored, {len(triggers)} triggers")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fundamental":
        run_quarterly_fundamental()
    else:
        run_weekly_research()
