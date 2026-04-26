"""
Patch main.py to use V2 tiered jobs.
Run: python /tmp/patch_main_v2.py
"""

path = "/root/trading-engine/main.py"
content = open(path).read()

# 1. Update imports to use v2 jobs
old_imports = """from jobs.morning_scan import run as morning_scan
from jobs.intraday_loop import run as intraday_loop"""

new_imports = """from jobs.morning_scan_v2 import run as morning_scan
from jobs.intraday_loop_v2 import run as intraday_loop"""

if old_imports in content:
    content = content.replace(old_imports, new_imports)
    print("1. Imports updated to V2 jobs")
else:
    print("1. SKIP — imports already updated or different")

# 2. Add watchlist build to startup
old_startup = '''    log.info("Scheduler running. Press Ctrl+C to stop.")'''
new_startup = '''    # Build tiered watchlist on startup
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

    log.info("Scheduler running. Press Ctrl+C to stop.")'''

if old_startup in content:
    content = content.replace(old_startup, new_startup)
    print("2. Startup watchlist build added")
else:
    print("2. SKIP — startup already updated")

# 3. Add --watchlist CLI command
old_cli = '    elif args.triggers:'
new_cli = """    elif args.watchlist:
        from core.auth import get_kite
        from core.watchlist import WatchlistManager
        kite = get_kite()
        if kite:
            mgr = WatchlistManager()
            mgr.refresh_instruments(kite)
            result = mgr.build_all()
            print(f"\\nTier 1: {len(result['tier1'])} instruments (active trading)")
            print(f"Tier 2: {len(result['tier2'])} instruments (daily scan)")
            print(f"Tier 3: {result['tier3_count']} stocks (fundamental universe)")
    elif args.triggers:"""

if old_cli in content:
    content = content.replace(old_cli, new_cli)
    print("3. --watchlist CLI command added")

# 4. Add argparse option
old_arg = '    parser.add_argument("--triggers"'
new_arg = '    parser.add_argument("--watchlist", action="store_true", help="Build tiered watchlists from Kite")\n    parser.add_argument("--triggers"'

if new_arg not in content:
    content = content.replace(old_arg, new_arg)
    print("4. --watchlist argument added")

open(path, "w").write(content)
print("\nmain.py patched successfully for V2 tiered system")
