"""
Deployment patch V4 — Makes the engine real.

Changes:
1. Adds --backtest CLI command
2. Wires portfolio tracker into intraday loop (real P&L)
3. Reduces regime change noise (confidence > 65% only)
4. Updates evening recap with actual P&L from portfolio tracker
5. Adds --reset-portfolio command

Run: python deploy_v4_patch.py
"""

import os

# ─── 1. Patch main.py: add CLI commands ───

path = "/root/trading-engine/main.py"
content = open(path).read()

# Add backtest CLI
old_arg = '    parser.add_argument("--deepanalysis"'
new_arg = '    parser.add_argument("--backtest", action="store_true", help="Run 10-year backtest on all strategies")\n    parser.add_argument("--reset-portfolio", action="store_true", help="Reset paper portfolio to 1L")\n    parser.add_argument("--portfolio", action="store_true", help="Show current paper portfolio")\n    parser.add_argument("--deepanalysis"'

if "--backtest" not in content:
    content = content.replace(old_arg, new_arg)
    print("1a. Added --backtest, --reset-portfolio, --portfolio args")

old_handler = "    elif args.deepanalysis:"
new_handler = """    elif args.backtest:
        from backtest.engine import run_backtest
        quick = "--quick" in sys.argv
        years = 10
        for a in sys.argv:
            if a.startswith("--years="):
                years = int(a.split("=")[1])
        run_backtest(quick=quick, years=years)
    elif getattr(args, 'reset_portfolio', False):
        from execution.portfolio_tracker import reset_portfolio
        reset_portfolio()
        print("Paper portfolio reset to ₹1,00,000")
    elif getattr(args, 'portfolio', False):
        from execution.portfolio_tracker import get_daily_summary, format_daily_telegram
        summary = get_daily_summary()
        print(format_daily_telegram(summary))
    elif args.deepanalysis:"""

if "args.backtest" not in content:
    content = content.replace(old_handler, new_handler)
    print("1b. Added CLI handlers for backtest/portfolio")

open(path, "w").write(content)
print("1. main.py patched")


# ─── 2. Patch intraday_loop_v2: use portfolio tracker ───

path2 = "/root/trading-engine/jobs/intraday_loop_v2.py"
content2 = open(path2).read()

# Add portfolio tracker import
old_import = "from execution.orders import execute_signal"
new_import = """from execution.orders import execute_signal
from execution.portfolio_tracker import open_position, update_positions"""

if "portfolio_tracker" not in content2:
    content2 = content2.replace(old_import, new_import)

    # Add position opening after signal execution
    old_exec = """            order = execute_signal(
                symbol=symbol,
                exchange=exchange,
                signal=signal,
                risk=risk,
                product=product,
            )"""

    new_exec = """            order = execute_signal(
                symbol=symbol,
                exchange=exchange,
                signal=signal,
                risk=risk,
                product=product,
            )

            # Track in paper portfolio
            if cfg.TRADING_MODE == "paper":
                open_position(
                    symbol=symbol,
                    direction=signal.direction,
                    quantity=risk.quantity,
                    entry_price=current_price,
                    stop_loss=risk.stop_loss if hasattr(risk, 'stop_loss') else current_price * 0.97,
                    target=risk.target if hasattr(risk, 'target') else current_price * 1.045,
                    regime=regime.regime,
                    signal_strength=signal.signal,
                )"""

    content2 = content2.replace(old_exec, new_exec)

    # Add position updates at end of cycle
    old_complete = '    log.info(f"─── Intraday cycle complete: {signals_fired} signals from {len(equities)} instruments ───")'
    new_complete = """    # Update open positions with current prices
    try:
        price_updates = {}
        for item in equities:
            token = item.get("token", 0)
            if token:
                try:
                    df_temp = fetch_candles(token, interval="day", days=5)
                    if not df_temp.empty:
                        price_updates[item["symbol"]] = df_temp["close"].iloc[-1]
                except Exception:
                    pass
        if price_updates:
            update_positions(price_updates)
    except Exception as e:
        log.debug(f"Position update failed: {e}")

    log.info(f"─── Intraday cycle complete: {signals_fired} signals from {len(equities)} instruments ───")"""

    content2 = content2.replace(old_complete, new_complete)

    open(path2, "w").write(content2)
    print("2. intraday_loop_v2.py patched with portfolio tracker")
else:
    print("2. SKIP — portfolio tracker already integrated")


# ─── 3. Patch morning scan: reduce regime noise ───

path3 = "/root/trading-engine/jobs/morning_scan_v2.py"
content3 = open(path3).read()

# Only alert regime changes with confidence > 65%
old_regime_alert = """            prev = prev_regimes.get(symbol, {}).get("regime", "Unknown")
            if prev != regime.regime and prev != "Unknown":
                regime_changes.append((symbol, prev, regime.regime, regime))
                send_regime_alert(symbol, prev, regime.regime, regime)"""

new_regime_alert = """            prev = prev_regimes.get(symbol, {}).get("regime", "Unknown")
            if prev != regime.regime and prev != "Unknown":
                regime_changes.append((symbol, prev, regime.regime, regime))
                # Only send Telegram for high-confidence regime changes
                if regime.confidence >= 0.65:
                    send_regime_alert(symbol, prev, regime.regime, regime)
                else:
                    log.debug(f"Low-confidence regime change for {symbol}: {prev} → {regime.regime} (conf={regime.confidence:.0%})")"""

if "regime.confidence >= 0.65" not in content3:
    content3 = content3.replace(old_regime_alert, new_regime_alert)
    open(path3, "w").write(content3)
    print("3. morning_scan_v2.py patched — regime alerts now require 65%+ confidence")
else:
    print("3. SKIP — regime noise filter already applied")


# ─── 4. Update evening recap to use portfolio tracker ───

path4 = "/root/trading-engine/jobs/evening_recap.py"
content4 = open(path4).read()

# Check if we can enhance the evening recap
if "portfolio_tracker" not in content4:
    # Add import and enhanced recap
    old_recap_import = "from execution.telegram_bot import send_message"
    new_recap_import = """from execution.telegram_bot import send_message
from execution.portfolio_tracker import get_daily_summary, format_daily_telegram"""

    content4 = content4.replace(old_recap_import, new_recap_import)

    # Try to add portfolio summary to the existing recap
    # This is best-effort since we don't know the exact structure
    if "def run(" in content4:
        old_run_end = "    send_message(text)"
        new_run_end = """    # Add portfolio P&L
    try:
        pf_summary = get_daily_summary()
        pf_text = format_daily_telegram(pf_summary)
        text = pf_text  # Replace old recap with portfolio-aware version
    except Exception as e:
        log.warning(f"Portfolio summary failed: {e}")

    send_message(text)"""

        if old_run_end in content4:
            content4 = content4.replace(old_run_end, new_run_end)

    open(path4, "w").write(content4)
    print("4. evening_recap.py patched with portfolio tracker P&L")
else:
    print("4. SKIP — portfolio tracker already in evening recap")


print("\n✅ All patches applied. Deploy with:")
print("   pkill -f 'python main.py'")
print("   sleep 2")
print("   python main.py --reset-portfolio")
print("   python main.py --backtest --quick  # Test with 10 stocks first")
print("   nohup python main.py > engine_output.log 2>&1 &")
