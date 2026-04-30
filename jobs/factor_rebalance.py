"""
jobs/factor_rebalance.py

Monthly factor rebalance job for the engine scheduler.

Add to main.py scheduler:
    schedule.every().day.at("14:30").do(factor_rebalance_check)

Runs daily at 2:30 PM but only executes on the last 2 trading
days of each month. This ensures the rebalance fires even if
the last day is a market holiday.
"""

from __future__ import annotations

import calendar
from datetime import datetime

from utils.logger import log


def run():
    """
    Check if today is rebalance day and execute if so.
    Called daily by scheduler — only acts on month-end.
    """
    today = datetime.now()
    last_day = calendar.monthrange(today.year, today.month)[1]

    # Only run on last 2 days of month (handles holidays)
    if today.day < last_day - 1:
        log.debug(f"Factor rebalance: not month-end (day {today.day}/{last_day}), skipping")
        return

    log.info("═══ FACTOR ENGINE — MONTHLY REBALANCE CHECK ═══")

    try:
        from factors.rebalancer import run_monthly_rebalance, format_rebalance_telegram
        from execution.telegram_bot import send_message

        # Run dry-run first (paper trading mode)
        summary = run_monthly_rebalance(
            use_kite=True,
            dry_run=False,   # Set to True if you want manual approval
            capital=None,    # Uses portfolio state
        )

        if "error" in summary:
            log.error(f"Factor rebalance failed: {summary['error']}")
            send_message(f"⚠️ Factor rebalance failed: {summary['error']}")
            return

        # Send Telegram summary
        msg = format_rebalance_telegram(summary)
        send_message(msg)

        log.info(
            f"Factor rebalance complete | "
            f"Portfolio: ₹{summary['portfolio_value']:,.0f} | "
            f"Buys: {summary['buys']} | Sells: {summary['sells']} | "
            f"Turnover: {summary['turnover_pct']:.1f}%"
        )

    except Exception as e:
        log.error(f"Factor rebalance error: {e}")
        try:
            from execution.telegram_bot import send_message
            send_message(f"⚠️ Factor rebalance error: {e}")
        except Exception:
            pass


def force_run():
    """Force a rebalance regardless of date. For testing."""
    log.info("═══ FACTOR ENGINE — FORCED REBALANCE ═══")

    from factors.rebalancer import run_monthly_rebalance, format_rebalance_telegram

    summary = run_monthly_rebalance(
        use_kite=True,
        dry_run=True,   # Always dry-run on force
        capital=None,
    )

    if "error" not in summary:
        msg = format_rebalance_telegram(summary)
        print(msg)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebalance (dry run)")
    parser.add_argument("--live", action="store_true", help="Force + execute")
    args = parser.parse_args()

    if args.live:
        from factors.rebalancer import run_monthly_rebalance, format_rebalance_telegram
        summary = run_monthly_rebalance(use_kite=True, dry_run=False)
        if "error" not in summary:
            print(format_rebalance_telegram(summary))
    elif args.force:
        force_run()
    else:
        run()
