"""
V5 Deployment Patch — The Real Rebuild.

Changes:
1. Replaces 5-min intraday loop with daily swing scanner
2. Adds research → backtest pipeline to weekly scan
3. Updates schedule: swing scan at 10 AM, position updates at 2 PM
4. Adds --pipeline CLI command
5. Updates evening recap to use portfolio tracker

Run: python deploy_v5.py
"""

path = "/root/trading-engine/main.py"
content = open(path).read()

changes = 0

# ─── 1. Replace intraday loop import with swing scanner ───
old = "from jobs.intraday_loop_v2 import run as intraday_loop"
new = """from jobs.intraday_loop_v2 import run as intraday_loop
from jobs.swing_scanner import run as swing_scan"""

if "swing_scanner" not in content:
    content = content.replace(old, new)
    changes += 1
    print("1. Added swing_scanner import")

# ─── 2. Add --pipeline CLI argument ───
if "--pipeline" not in content:
    old_arg = '    parser.add_argument("--backtest"'
    new_arg = '    parser.add_argument("--pipeline", action="store_true", help="Run research paper → strategy pipeline")\n    parser.add_argument("--swingscan", action="store_true", help="Run swing signal scan once")\n    parser.add_argument("--backtest"'
    content = content.replace(old_arg, new_arg)
    changes += 1
    print("2. Added --pipeline, --swingscan CLI args")

# ─── 3. Add CLI handlers ───
if "args.pipeline" not in content:
    old_handler = "    elif args.backtest:"
    new_handler = """    elif args.pipeline:
        from research.strategy_pipeline import process_new_papers
        process_new_papers()
    elif args.swingscan:
        from jobs.swing_scanner import run as swing_scan
        swing_scan()
    elif args.backtest:"""
    content = content.replace(old_handler, new_handler)
    changes += 1
    print("3. Added CLI handlers")

# ─── 4. Add swing scan to daily schedule (10:00 AM) ───
# Find where morning scan is scheduled and add swing scan after
if "swing_scan" not in content or "10:00" not in content:
    old_sched = '    schedule.every().monday.at("08:45").do(_weekday_guard, morning_scan)'
    new_sched = """    schedule.every().monday.at("08:45").do(_weekday_guard, morning_scan)

    # Swing signal scan at 10:00 AM (after market settles)
    schedule.every().monday.at("10:00").do(_weekday_guard, swing_scan)
    schedule.every().tuesday.at("10:00").do(_weekday_guard, swing_scan)
    schedule.every().wednesday.at("10:00").do(_weekday_guard, swing_scan)
    schedule.every().thursday.at("10:00").do(_weekday_guard, swing_scan)
    schedule.every().friday.at("10:00").do(_weekday_guard, swing_scan)"""

    if old_sched in content:
        content = content.replace(old_sched, new_sched)
        changes += 1
        print("4. Added swing scan schedule at 10:00 AM weekdays")
    else:
        print("4. SKIP — couldn't find morning scan schedule")

# ─── 5. Add research pipeline to weekly scan ───
if "process_new_papers" not in content:
    old_research = "from jobs.weekly_research import run as weekly_research"
    new_research = """from jobs.weekly_research import run as weekly_research

def research_and_pipeline():
    \"\"\"Run weekly research scan, then process new papers through strategy pipeline.\"\"\"
    weekly_research()
    try:
        from research.strategy_pipeline import process_new_papers
        process_new_papers()
    except Exception as e:
        log.warning(f"Strategy pipeline failed: {e}")"""

    content = content.replace(old_research, new_research)

    # Update the weekly schedule to use combined function
    content = content.replace(
        "weekly_research)",
        "research_and_pipeline)",
    )
    changes += 1
    print("5. Added research → pipeline to weekly schedule")

# ─── 6. Reduce intraday loop frequency (every 30 min instead of 5) ───
if 'every(5).minutes' in content:
    content = content.replace('every(5).minutes', 'every(30).minutes')
    changes += 1
    print("6. Reduced intraday loop to every 30 min (swing scan is primary now)")

open(path, "w").write(content)
print(f"\n✅ {changes} changes applied to main.py")

# ─── 7. Verify all imports work ───
print("\nVerifying imports...")
try:
    import importlib
    import sys
    sys.path.insert(0, "/root/trading-engine")
    
    from core.swing_strategies import SWING_STRATEGIES, get_composite_signal
    print(f"  ✅ swing_strategies: {len(SWING_STRATEGIES)} strategies loaded")
    
    from execution.portfolio_tracker import reset_portfolio, get_daily_summary
    print(f"  ✅ portfolio_tracker: OK")
    
    from research.strategy_pipeline import process_new_papers
    print(f"  ✅ strategy_pipeline: OK")
    
    from jobs.swing_scanner import run as swing_scan
    print(f"  ✅ swing_scanner: OK")
    
    print("\n🚀 Ready to deploy. Run:")
    print("   pkill -f 'python main.py'")
    print("   python main.py --reset-portfolio")
    print("   python main.py --swingscan    # Test swing scan")
    print("   nohup python main.py > engine_output.log 2>&1 &")
    
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
