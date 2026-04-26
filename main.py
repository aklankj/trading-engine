#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  REGIME-ADAPTIVE TRADING ENGINE — Main Orchestrator
═══════════════════════════════════════════════════════════════

Entry point that schedules and runs all jobs:

  Weekday schedule (IST):
    08:45  Morning scan — regime detection, pre-market summary
    09:15  Intraday loop starts (every 5 min until 15:30)
    15:00  Fundamental buy trigger check
    16:00  Evening recap — P&L, signals, strategy performance

  Weekly:
    Sunday 20:00  Research paper scan

  Quarterly:
    Manual trigger  Fundamental quality rescan

Usage:
  python main.py                  # Run scheduler (production)
  python main.py --morning        # Run morning scan once
  python main.py --intraday       # Run one intraday cycle
  python main.py --evening        # Run evening recap once
  python main.py --research       # Run research scan once
  python main.py --fundamental    # Run fundamental rescan once
  python main.py --status         # Print system status
"""

import sys
import time
import signal
import argparse
import schedule
from datetime import datetime

from config.settings import cfg
from utils.logger import log
from utils import is_weekday, is_market_hours, now_ist

# ── Import all jobs ───────────────────────────────────────────
from jobs.morning_scan_v2 import run as morning_scan
from jobs.swing_scanner_v2 import run as swing_scan_v2, update_open_positions
# OLD: # OLD: from jobs.intraday_loop_v2 import run as intraday_loop  # KILLED IN V6  # KILLED IN V6
# OLD: # OLD: from jobs.swing_scanner import run as swing_scan  # KILLED IN V6  # KILLED IN V6
from jobs.evening_recap import run as evening_recap
from jobs.weekly_research import run_weekly_research, run_quarterly_fundamental
from fundamental.buy_triggers import check_buy_triggers
from core.auto_auth import auto_login


def setup_schedule():
    """Configure the job schedule."""

    # ── Weekday jobs (IST times) ──────────────────────────────
    schedule.every().monday.at("08:45").do(_weekday_guard, morning_scan)

    # Swing signal scan at 10:00 AM (after market settles)
    # KILLED V6: # KILLED V6: schedule.every().monday.at("10:00").do(_weekday_guard, swing_scan)
    # KILLED V6: # KILLED V6: schedule.every().tuesday.at("10:00").do(_weekday_guard, swing_scan)
    # KILLED V6: # KILLED V6: schedule.every().wednesday.at("10:00").do(_weekday_guard, swing_scan)
    # KILLED V6: # KILLED V6: schedule.every().thursday.at("10:00").do(_weekday_guard, swing_scan)
    # KILLED V6: # KILLED V6: schedule.every().friday.at("10:00").do(_weekday_guard, swing_scan)
    schedule.every().tuesday.at("08:45").do(_weekday_guard, morning_scan)
    schedule.every().wednesday.at("08:45").do(_weekday_guard, morning_scan)
    schedule.every().thursday.at("08:45").do(_weekday_guard, morning_scan)
    schedule.every().friday.at("08:45").do(_weekday_guard, morning_scan)

    # Intraday loop every 5 minutes (market hours guard is inside the job)
    # KILLED V6: # KILLED V6: schedule.every(30).minutes.do(_weekday_guard, intraday_loop)

    # Fundamental buy trigger check at 3:00 PM
    schedule.every().monday.at("15:00").do(_weekday_guard, check_buy_triggers)
    schedule.every().tuesday.at("15:00").do(_weekday_guard, check_buy_triggers)
    schedule.every().wednesday.at("15:00").do(_weekday_guard, check_buy_triggers)
    schedule.every().thursday.at("15:00").do(_weekday_guard, check_buy_triggers)
    schedule.every().friday.at("15:00").do(_weekday_guard, check_buy_triggers)

    # Evening recap at 4:00 PM
    schedule.every().monday.at("16:00").do(_weekday_guard, evening_recap)
    schedule.every().tuesday.at("16:00").do(_weekday_guard, evening_recap)
    schedule.every().wednesday.at("16:00").do(_weekday_guard, evening_recap)
    schedule.every().thursday.at("16:00").do(_weekday_guard, evening_recap)
    schedule.every().friday.at("16:00").do(_weekday_guard, evening_recap)

    # ── Weekly jobs ───────────────────────────────────────────
    schedule.every().sunday.at("20:00").do(run_weekly_research)
    # Run strategy pipeline 30 min after research scan
    def _run_pipeline():
        try:
            from research.strategy_pipeline import process_new_papers
            process_new_papers()
        except Exception as e:
            log.warning(f"Strategy pipeline failed: {e}")
    schedule.every().sunday.at("20:30").do(_run_pipeline)

    # Auto-login at 6:30 AM IST (tokens expire at ~6 AM)
    schedule.every().monday.at("06:30").do(auto_login)
    schedule.every().tuesday.at("06:30").do(auto_login)
    schedule.every().wednesday.at("06:30").do(auto_login)
    schedule.every().thursday.at("06:30").do(auto_login)
    schedule.every().friday.at("06:30").do(auto_login)

    log.info("Schedule configured — all jobs registered (including auto-login)")


def _weekday_guard(job_fn):
    """Only run job on weekdays."""
    if is_weekday():
        try:
            job_fn()
        except Exception as e:
            log.error(f"Job {job_fn.__module__}.{job_fn.__name__} failed: {e}")
    else:
        log.debug(f"Skipping {job_fn.__name__} — weekend")


def run_scheduler():
    """Main loop — runs forever, executing scheduled jobs."""
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   REGIME-ADAPTIVE TRADING ENGINE                    ║")
    log.info(f"║   Mode: {cfg.TRADING_MODE:12s}                           ║")
    log.info(f"║   Capital: ₹{cfg.INITIAL_CAPITAL:>10,.0f}                        ║")
    log.info(f"║   Watchlist: {len(cfg.WATCHLIST):>3d} instruments                    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # Validate config
    issues = cfg.validate()
    if issues:
        for issue in issues:
            log.warning(f"Config issue: {issue}")
        if "KITE_API_KEY" in str(issues):
            log.error("Cannot start without Kite API credentials")
            sys.exit(1)

    setup_schedule()

    # Graceful shutdown
    def _shutdown(signum, frame):
        log.info("Shutdown signal received — stopping engine")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Build tiered watchlist on startup
    try:
        from core.auth import get_kite
        from core.watchlist import WatchlistManager
        kite = get_kite()
        if kite:
            mgr = WatchlistManager()
            mgr.refresh_instruments(kite)
            result = mgr.build_all()
            log.info(f"Watchlists: T1={len(result['tier1'])}, T2={len(result['tier2'])}, T3={result['tier3_count']}")
    except Exception as e:
        log.warning(f"Watchlist init failed (will retry at morning scan): {e}")

    log.info("Scheduler running. Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt — stopping engine")
            break
        except Exception as e:
            log.error(f"Scheduler error: {e}")
            time.sleep(60)


def print_status():
    """Print current system status."""
    from utils import load_json, fmt_inr

    print("\n═══ TRADING ENGINE STATUS ═══\n")
    print(f"  Mode:        {cfg.TRADING_MODE}")
    print(f"  Capital:     {fmt_inr(cfg.INITIAL_CAPITAL)}")
    print(f"  Watchlist:   {len(cfg.WATCHLIST)} instruments")

    issues = cfg.validate()
    if issues:
        print(f"\n  ⚠️  Config issues:")
        for i in issues:
            print(f"     - {i}")
    else:
        print(f"\n  ✅ Config valid")

    trade_log = load_json(cfg.TRADE_LOG, default={"trades": []})
    print(f"\n  Trades logged: {len(trade_log.get('trades', []))}")

    paper_log = load_json(cfg.PAPER_LOG, default={"papers": []})
    print(f"  Papers tracked: {len(paper_log.get('papers', []))}")

    regime_state = load_json(cfg.DATA_DIR / "regime_state.json", default={})
    if regime_state:
        print(f"\n  Latest regimes:")
        for sym, data in list(regime_state.items())[:5]:
            print(f"    {sym:15s} → {data.get('regime', 'Unknown'):10s} (conf: {data.get('confidence', 0):.0%})")

    print()


# ── CLI ───────────────────────────────────────────────────────

def main():
    # Prevent duplicate instances
    import os
    pidfile = cfg.DATA_DIR / "engine.pid"
    if pidfile.exists():
        old_pid = int(pidfile.read_text().strip())
        try:
            os.kill(old_pid, 0)  # Check if process exists
            log.warning(f"Engine already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except ProcessLookupError:
            pass  # Old process dead, continue
    pidfile.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: pidfile.unlink(missing_ok=True))
    parser = argparse.ArgumentParser(description="Regime-Adaptive Trading Engine")
    parser.add_argument("--morning", action="store_true", help="Run morning scan once")
    parser.add_argument("--intraday", action="store_true", help="Run one intraday cycle")
    parser.add_argument("--evening", action="store_true", help="Run evening recap once")
    parser.add_argument("--research", action="store_true", help="Run research scan once")
    parser.add_argument("--fundamental", action="store_true", help="Run fundamental rescan")
    parser.add_argument("--watchlist", action="store_true", help="Build tiered watchlists from Kite")
    parser.add_argument("--triggers", action="store_true", help="Check buy triggers")
    parser.add_argument("--fullscan", action="store_true", help="Scan ALL NSE stocks with news analysis")
    parser.add_argument("--fundscan", action="store_true", help="Run full fundamental scan")
    parser.add_argument("--status", action="store_true", help="Print system status")
    parser.add_argument("--pipeline", action="store_true", help="Run research paper → strategy pipeline")
    parser.add_argument("--swingscan", action="store_true", help="Run swing signal scan once")
    parser.add_argument("--backtest", action="store_true", help="Run 10-year backtest")
    parser.add_argument("--opportunities", action="store_true", help="Bear market opportunity scan")
    parser.add_argument("--deepanalysis", action="store_true", help="AI deep analysis of top picks")
    parser.add_argument("--reset-portfolio", action="store_true", dest="reset_portfolio", help="Reset paper portfolio")
    parser.add_argument("--portfolio", action="store_true", help="Show paper portfolio")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.morning:
        morning_scan()
    elif args.intraday:
        intraday_loop()
    elif args.evening:
        evening_recap()
    elif args.research:
        run_weekly_research()
        try:
            from research.strategy_pipeline import process_new_papers
            process_new_papers()
        except Exception as e:
            print(f"Pipeline failed: {e}")
    elif args.fullscan:
        from fundamental.full_universe_scan import run_full_universe_scan
        n = None
        for a in sys.argv:
            if a.isdigit(): n = int(a)
        run_full_universe_scan(max_stocks=n, skip_news="--no-news" in sys.argv)
    elif args.fundscan:
        from fundamental.screener_scraper import run_full_fundamental_scan
        run_full_fundamental_scan()
    elif args.fundamental:
        run_quarterly_fundamental()
    elif args.watchlist:
        from core.auth import get_kite
        from core.watchlist import WatchlistManager
        kite = get_kite()
        if kite:
            mgr = WatchlistManager()
            mgr.refresh_instruments(kite)
            result = mgr.build_all()
            print(f"\nTier 1: {len(result['tier1'])} instruments (active trading)")
            print(f"Tier 2: {len(result['tier2'])} instruments (daily scan)")
            print(f"Tier 3: {result['tier3_count']} stocks (fundamental universe)")
    elif args.triggers:
        check_buy_triggers()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
