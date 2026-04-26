"""
V6 Deployment — The Clean Rebuild.

1. Kills the old intraday loop (meta_composite) completely
2. Removes old strategy imports
3. Wires swing_scanner_v2 as the ONLY signal source
4. Updates CLI to use unified backtester
5. Adds position update job at 2:30 PM
6. Cleans schedule

Run: python deploy_v6.py
"""

path = "/root/trading-engine/main.py"
content = open(path).read()

changes = 0

# ─── 1. Replace old intraday loop with swing scanner v2 ───
old_imports = [
    "from jobs.intraday_loop_v2 import run as intraday_loop",
    "from jobs.intraday_loop import run as intraday_loop",
    "from jobs.swing_scanner import run as swing_scan",
]
for old in old_imports:
    if old in content:
        content = content.replace(old, "# OLD: " + old + "  # KILLED IN V6")
        changes += 1

# Add new import if not present
new_import = "from jobs.swing_scanner_v2 import run as swing_scan_v2, update_open_positions"
if "swing_scanner_v2" not in content:
    # Insert after the morning scan import
    anchor = "from jobs.morning_scan_v2 import run as morning_scan"
    if anchor in content:
        content = content.replace(anchor, anchor + "\n" + new_import)
        changes += 1
        print("1. Added swing_scanner_v2 import")

# ─── 2. Kill all old intraday schedules ───
# Remove every(5).minutes and every(30).minutes intraday calls
lines = content.split("\n")
new_lines = []
for line in lines:
    if "intraday_loop" in line and "schedule" in line:
        new_lines.append("    # KILLED V6: " + line.strip())
        changes += 1
    elif "every(5).minutes" in line or "every(30).minutes" in line:
        new_lines.append("    # KILLED V6: " + line.strip())
        changes += 1
    else:
        new_lines.append(line)
content = "\n".join(new_lines)
print("2. Killed old intraday loop schedules")

# ─── 3. Add swing scan v2 schedule (10 AM) if not present ───
if "swing_scan_v2" not in content or "10:00" not in content:
    # Find the morning scan schedule and add after it
    old_morning = '    schedule.every().monday.at("08:45").do(_weekday_guard, morning_scan)'
    new_morning = """    schedule.every().monday.at("08:45").do(_weekday_guard, morning_scan)

    # Swing signal scan at 10:00 AM (ONLY signal source)
    schedule.every().monday.at("10:00").do(_weekday_guard, swing_scan_v2)
    schedule.every().tuesday.at("10:00").do(_weekday_guard, swing_scan_v2)
    schedule.every().wednesday.at("10:00").do(_weekday_guard, swing_scan_v2)
    schedule.every().thursday.at("10:00").do(_weekday_guard, swing_scan_v2)
    schedule.every().friday.at("10:00").do(_weekday_guard, swing_scan_v2)

    # Position update at 2:30 PM (check stops/targets)
    schedule.every().monday.at("14:30").do(_weekday_guard, update_open_positions)
    schedule.every().tuesday.at("14:30").do(_weekday_guard, update_open_positions)
    schedule.every().wednesday.at("14:30").do(_weekday_guard, update_open_positions)
    schedule.every().thursday.at("14:30").do(_weekday_guard, update_open_positions)
    schedule.every().friday.at("14:30").do(_weekday_guard, update_open_positions)"""

    if old_morning in content:
        content = content.replace(old_morning, new_morning)
        changes += 1
        print("3. Added swing_scan_v2 (10 AM) + position update (2:30 PM)")
    else:
        print("3. WARN — couldn't find morning scan schedule anchor")

# ─── 4. Remove old swing_scan schedules (avoid duplicates) ───
lines = content.split("\n")
new_lines = []
skip_next = False
for i, line in enumerate(lines):
    # Kill old swing_scan (not v2) schedules
    if "swing_scan)" in line and "swing_scan_v2" not in line and "schedule" in line:
        new_lines.append("    # KILLED V6: " + line.strip())
        changes += 1
    else:
        new_lines.append(line)
content = "\n".join(new_lines)
print("4. Removed old swing_scan schedules")

# ─── 5. Update backtest CLI to use unified backtester ───
old_bt = """    elif args.backtest:
        from backtest.engine import run_backtest
        quick = "--quick" in sys.argv
        years = 10
        for a in sys.argv:
            if a.startswith("--years="):
                years = int(a.split("=")[1])
        run_backtest(quick=quick, years=years)"""

new_bt = """    elif args.backtest:
        from backtest.unified import run_backtest, TOP10, NIFTY50, NIFTY100_PLUS
        n100 = "--nifty100" in sys.argv
        full = "--full" in sys.argv
        wf = "--walkforward" in sys.argv
        years = 10
        for a in sys.argv:
            if a.startswith("--years="):
                years = int(a.split("=")[1])
        if n100:
            symbols = NIFTY100_PLUS
        elif full:
            symbols = NIFTY50
        else:
            symbols = TOP10
        run_backtest(symbols=symbols, years=years, walk_forward=wf)"""

if old_bt in content:
    content = content.replace(old_bt, new_bt)
    changes += 1
    print("5. Updated --backtest to use unified backtester")
else:
    print("5. SKIP — backtest CLI format different")

# ─── 6. Update --swingscan CLI ───
old_swing_cli = """    elif args.swingscan:
        from jobs.swing_scanner import run as swing_scan
        swing_scan()"""

new_swing_cli = """    elif args.swingscan:
        from jobs.swing_scanner_v2 import run as swing_scan_v2
        swing_scan_v2()"""

if old_swing_cli in content:
    content = content.replace(old_swing_cli, new_swing_cli)
    changes += 1
    print("6. Updated --swingscan CLI to v2")

open(path, "w").write(content)

print(f"\n✅ {changes} changes applied")
print(f"\n📋 New schedule:")
print(f"  06:30 — Auto Kite login")
print(f"  08:45 — Morning regime scan")
print(f"  10:00 — Swing signal scan (ONLY signal source)")
print(f"  14:30 — Position update (check stops/targets)")
print(f"  16:00 — Evening recap (real P&L)")
print(f"  Sunday 20:00 — Research scan")
print(f"  Sunday 20:30 — Strategy pipeline (backtest new papers)")
print(f"\n🚫 KILLED:")
print(f"  - Old intraday loop (meta_composite)")
print(f"  - Old swing_scanner v1")
print(f"  - Every 5/30 minute scanning")
print(f"\nDeploy:")
print(f"  python -c 'from execution.portfolio_tracker import reset_portfolio; reset_portfolio()'")
print(f"  python -m backtest.unified                  # Verify strategies work")
print(f"  python main.py --swingscan                 # Test live signals")
print(f"  nohup python main.py > engine_output.log 2>&1 &")
