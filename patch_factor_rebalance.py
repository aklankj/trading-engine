"""
patch_factor_rebalance.py

Adds the factor rebalance job to main.py scheduler.
Run once: python patch_factor_rebalance.py

Adds:
  - 14:30 daily job that checks if it's month-end and rebalances
  - --factor-rebalance CLI command for manual runs
  - --factor-dry-run CLI command for dry runs
"""

path = "/root/trading-engine/main.py"
content = open(path).read()
patched = False

# 1. Add factor rebalance import
import_marker = "from jobs.evening_recap import run as evening_recap"
factor_import = "from jobs.factor_rebalance import run as factor_rebalance_check"

if factor_import not in content and import_marker in content:
    content = content.replace(
        import_marker,
        f"{import_marker}\n{factor_import}",
    )
    print("1. Added factor_rebalance import")
    patched = True
else:
    print("1. SKIP — import already present or marker not found")

# 2. Add schedule entry (after evening recap schedule)
schedule_marker = 'schedule.every().day.at("16:00")'
factor_schedule = '    schedule.every().day.at("14:30").do(factor_rebalance_check)  # Factor engine monthly'

if "factor_rebalance_check" not in content and schedule_marker in content:
    # Find the line with the evening recap schedule and add after it
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if schedule_marker in line and "evening" in line.lower() or (schedule_marker in line and "recap" in content[content.index(line):content.index(line)+100].lower() if line in content else False):
            new_lines.append(factor_schedule)
    
    # Simpler approach: just add after the schedule_marker line
    if factor_schedule.strip() not in '\n'.join(new_lines):
        idx = content.index(schedule_marker)
        line_end = content.index('\n', idx)
        content = content[:line_end+1] + factor_schedule + '\n' + content[line_end+1:]
    
    print("2. Added 14:30 factor rebalance schedule")
    patched = True
else:
    print("2. SKIP — schedule already present or marker not found")

# 3. Add CLI commands
cli_marker = '    elif args.evening:'
factor_cli = '''    elif args.factor_rebalance:
        from jobs.factor_rebalance import force_run
        force_run()
    elif args.factor_live:
        from factors.rebalancer import run_monthly_rebalance, format_rebalance_telegram
        summary = run_monthly_rebalance(use_kite=True, dry_run=False)
        if "error" not in summary:
            print(format_rebalance_telegram(summary))'''

if "factor_rebalance" not in content and cli_marker in content:
    content = content.replace(cli_marker, factor_cli + '\n' + cli_marker)
    print("3. Added --factor-rebalance and --factor-live CLI commands")
    patched = True
else:
    print("3. SKIP — CLI already present or marker not found")

# 4. Add argparse options
argparse_marker = '    parser.add_argument("--evening"'
factor_args = '''    parser.add_argument("--factor-rebalance", action="store_true", help="Dry-run factor rebalance")
    parser.add_argument("--factor-live", action="store_true", help="Execute factor rebalance")'''

if "--factor-rebalance" not in content and argparse_marker in content:
    content = content.replace(argparse_marker, factor_args + '\n' + argparse_marker)
    print("4. Added argparse options")
    patched = True
else:
    print("4. SKIP — argparse already present or marker not found")

if patched:
    open(path, "w").write(content)
    print("\n✅ main.py patched for factor rebalance")
else:
    print("\n⚠️  Nothing to patch — already up to date or markers not found")
    print("    You can manually add to main.py:")
    print('    schedule.every().day.at("14:30").do(factor_rebalance_check)')
