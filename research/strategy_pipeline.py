"""
Research Paper → Strategy Pipeline.

When the weekly research scan finds a new paper:
1. OpenRouter extracts the core trading logic
2. Generates a Python strategy function
3. Backtests on 10 years of data
4. If Sharpe > 1.0 and PF > 1.5 → adds to paper trading
5. After 30 days of paper trading → report on performance
6. If still profitable → promote to approval mode

This is how the engine gets smarter over time.
"""

import re
import json
import time
import importlib
import sys
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import cfg
from backtest.engine import fetch_yfinance_data, TOP10_SYMBOLS
from utils.logger import log
from utils import load_json, save_json


STRATEGIES_DIR = cfg.DATA_DIR / "generated_strategies"
PIPELINE_LOG = cfg.DATA_DIR / "strategy_pipeline.json"

# Thresholds for promotion
BACKTEST_SHARPE_MIN = 1.0
BACKTEST_PF_MIN = 1.5
BACKTEST_WIN_RATE_MIN = 45.0
PAPER_TRADE_DAYS_MIN = 30


def extract_strategy_from_paper(paper: dict) -> dict:
    """
    Use OpenRouter to extract trading logic from a research paper
    and generate a backtestable Python function.
    """
    if not cfg.OPENROUTER_API_KEY:
        return {"error": "No API key"}

    title = paper.get("title", "")
    abstract = paper.get("summary", paper.get("abstract", ""))
    sharpe = paper.get("sharpe", "not reported")

    prompt = f"""You are a quantitative trading researcher. Extract the core trading strategy from this paper and implement it as a Python function.

PAPER: {title}
ABSTRACT: {abstract}
REPORTED SHARPE: {sharpe}

Generate a Python function with EXACTLY this signature:

def strategy(df):
    \"\"\"
    [One-line description of the strategy]
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    \"\"\"
    trades = []
    equity = 100000
    position = None
    
    # Your implementation here using only pandas and numpy
    # Use simple indicators: SMA, RSI, ATR, Bollinger Bands
    # Do NOT use any external libraries beyond pandas/numpy
    
    return trades, equity

CRITICAL RULES:
1. Use ONLY pandas and numpy operations
2. Start with equity = 100000
3. Position size = 10% of equity per trade
4. Must have stop loss and take profit logic
5. Return list of percent returns per trade, and final equity
6. Handle edge cases (empty df, insufficient data)
7. Minimum 200 rows of warmup before first trade
8. The function must be completely self-contained

CRITICAL CODING RULES:
1. ALWAYS use .iloc[i] for positional indexing, NEVER df['col'][i] (DatetimeIndex breaks this)
2. Returns must be in PERCENT (e.g., +5.2 means +5.2% gain)
3. Helper functions must be INSIDE the strategy() function
4. All indicator calculations must use .iloc for row access in loops
5. Test: the function must generate at least 20 trades on 10 years of daily data

Example of CORRECT row access:
    current_price = df['close'].iloc[i]   # CORRECT
    current_price = df['close'][i]         # WRONG - breaks with DatetimeIndex

Respond with ONLY the Python code, no explanation, no markdown backticks."""

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
                "max_tokens": 2000,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            return {"error": f"API returned {resp.status_code}"}

        code = resp.json()["choices"][0]["message"]["content"]
        # Clean up code
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        code = code.strip()

        return {
            "paper_title": title,
            "code": code,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e)[:200]}


def backtest_generated_strategy(code: str, strategy_name: str, years: int = 10) -> dict:
    """
    Backtest a generated strategy function on historical data.
    Returns performance metrics.
    """
    # Create a temporary module from the code
    try:
        # Add numpy/pandas imports if not present
        if "import numpy" not in code:
            code = "import numpy as np\nimport pandas as pd\n" + code

        # Fix common LLM code issues before executing
        # Replace df['col'][i] with df['col'].iloc[i] pattern
        import re as _re
        code = _re.sub(r"df\['(\w+)'\]\[(\w+)\]", r"df[''].iloc[]", code)
        code = _re.sub(r'df\["(\w+)"\]\[(\w+)\]', r'df[""].iloc[]', code)

        # Execute the full code (including helper functions) in one namespace
        exec_globals = {"np": np, "pd": __import__("pandas"), "numpy": np, "pandas": __import__("pandas")}
        # Finding #5: Restricted exec — no file/network/os access
        restricted_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in ("open", "exec", "eval", "compile", "__import__",
                         "breakpoint", "exit", "quit", "input")
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) for k in dir(__builtins__)
            if k not in ("open", "exec", "eval", "compile", "__import__",
                         "breakpoint", "exit", "quit", "input")
        }
        exec_globals["__builtins__"] = restricted_builtins
        exec(code, exec_globals)

        # Find the strategy function
        strategy_func = None
        for name, obj in exec_globals.items():
            if callable(obj) and name == "strategy":
                strategy_func = obj
                break

        if strategy_func is None:
            # Try wrapping all code into a single callable
            try:
                wrapper_code = code + "\n\n_result = strategy(df_input)"
                def run_strategy(df):
                    local_vars = {"df_input": df, "np": np, "pd": __import__('pandas')}
                    exec(code + "\n_trades, _equity = strategy(df_input)", {**exec_globals, **local_vars}, local_vars)
                    return local_vars.get("_trades", []), local_vars.get("_equity", 100000)
                strategy_func = run_strategy
            except Exception:
                return {"error": "No 'strategy' function found in generated code"}

    except Exception as e:
        return {"error": f"Code execution failed: {str(e)[:200]}"}

    # Run on test stocks
    all_trades = []
    all_equity = []
    errors = 0

    for sym in TOP10_SYMBOLS[:5]:  # Test on 5 stocks for speed
        try:
            df = fetch_yfinance_data(sym, years=years)
            if df.empty or len(df) < 500:
                continue

            trades, final_eq = strategy_func(df)

            if isinstance(trades, list) and len(trades) > 0:
                all_trades.extend(trades)
                all_equity.append(final_eq)
        except Exception as e:
            errors += 1
            log.debug(f"  {sym} failed: {e}")

        time.sleep(0.3)

    if not all_trades or len(all_trades) < 5:
        return {"error": f"Too few trades ({len(all_trades)}) on test stocks"}

    # Calculate metrics
    wins = [t for t in all_trades if t > 0]
    losses = [t for t in all_trades if t <= 0]
    win_rate = len(wins) / len(all_trades) * 100
    avg_return = np.mean([(e / 100000 - 1) * 100 for e in all_equity]) if all_equity else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    sharpe = 0
    if len(all_trades) > 1 and np.std(all_trades) > 0:
        sharpe = np.mean(all_trades) / np.std(all_trades) * np.sqrt(12)

    result = {
        "strategy_name": strategy_name,
        "total_trades": len(all_trades),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 1),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(np.mean(wins), 1) if wins else 0,
        "avg_loss": round(np.mean(losses), 1) if losses else 0,
        "errors": errors,
        "tested_on": len(all_equity),
        "years": years,
    }

    # Verdict
    if sharpe > BACKTEST_SHARPE_MIN and pf > BACKTEST_PF_MIN and win_rate > BACKTEST_WIN_RATE_MIN:
        result["verdict"] = "PROMOTED"
        result["action"] = "Add to paper trading"
    elif sharpe > 0.5 and pf > 1.2:
        result["verdict"] = "WATCHLIST"
        result["action"] = "Monitor, re-test with more data"
    else:
        result["verdict"] = "REJECTED"
        result["action"] = "Does not meet minimum thresholds"

    return result


def save_strategy(code: str, name: str, paper_title: str, backtest_result: dict):
    """Save a generated strategy to disk for future use."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:50]
    path = STRATEGIES_DIR / f"{safe_name}.py"
    path.write_text(code)

    # Update pipeline log
    pipeline = load_json(PIPELINE_LOG, default={"strategies": []})
    pipeline["strategies"].append({
        "name": name,
        "paper_title": paper_title,
        "file": str(path),
        "backtest": backtest_result,
        "status": backtest_result.get("verdict", "UNKNOWN"),
        "created_at": datetime.now().isoformat(),
        "paper_trade_start": None,
        "paper_trade_pnl": None,
        "promoted_to_live": False,
    })
    save_json(PIPELINE_LOG, pipeline)

    log.info(f"Strategy saved: {name} → {path}")


def process_new_papers():
    """
    Process all unprocessed research papers through the pipeline.
    Called after weekly research scan.
    """
    log.info("═══ RESEARCH → STRATEGY PIPELINE ═══")

    # Load paper log
    paper_log_data = load_json(cfg.DATA_DIR / "paper_log.json", default={"papers": []})
    paper_log = paper_log_data.get("papers", []) if isinstance(paper_log_data, dict) else paper_log_data
    pipeline = load_json(PIPELINE_LOG, default={"strategies": []})

    # Find unprocessed papers
    processed_titles = {s.get("paper_title", "") for s in pipeline.get("strategies", [])}
    new_papers = [p for p in paper_log if p.get("title", "") not in processed_titles]

    if not new_papers:
        log.info("No new papers to process")
        return

    log.info(f"Processing {len(new_papers)} new papers...")

    results = []
    for i, paper in enumerate(new_papers[:5]):  # Max 5 per run
        title = paper.get("title", "Unknown")[:60]
        log.info(f"\n  [{i+1}] {title}...")

        # Step 1: Extract strategy
        extracted = extract_strategy_from_paper(paper)
        if "error" in extracted:
            log.warning(f"    Extraction failed: {extracted['error']}")
            continue

        code = extracted["code"]
        name = re.sub(r"[^a-zA-Z0-9 ]", "", title)[:40].strip().replace(" ", "_")

        # Step 2: Backtest
        log.info(f"    Backtesting on 5 stocks, 10 years...")
        bt_result = backtest_generated_strategy(code, name)

        if "error" in bt_result:
            log.warning(f"    Backtest failed: {bt_result['error']}")
            # Save anyway for debugging
            save_strategy(code, name, title, {"error": bt_result["error"], "verdict": "ERROR"})
            continue

        log.info(
            f"    Results: Sharpe={bt_result['sharpe']:+.2f} PF={bt_result['profit_factor']:.2f} "
            f"WR={bt_result['win_rate']:.0f}% → {bt_result['verdict']}"
        )

        # Step 3: Save
        save_strategy(code, name, title, bt_result)
        results.append({"name": name, "title": title, **bt_result})

        time.sleep(2)

    # Report
    promoted = [r for r in results if r.get("verdict") == "PROMOTED"]
    watched = [r for r in results if r.get("verdict") == "WATCHLIST"]
    rejected = [r for r in results if r.get("verdict") == "REJECTED"]

    log.info(
        f"\n═══ PIPELINE COMPLETE ═══\n"
        f"  Processed: {len(results)}\n"
        f"  Promoted to paper trading: {len(promoted)}\n"
        f"  Watchlist: {len(watched)}\n"
        f"  Rejected: {len(rejected)}"
    )

    # Telegram report
    _send_pipeline_report(results, promoted, watched, rejected)

    return results


def _send_pipeline_report(all_results, promoted, watched, rejected):
    """Send pipeline results to Telegram."""
    try:
        from execution.telegram_bot import send_message

        msg = f"🔬 <b>Research → Strategy Pipeline</b>\n\n"
        msg += f"Processed: {len(all_results)} papers\n\n"

        if promoted:
            msg += "🟢 <b>PROMOTED to paper trading:</b>\n"
            for r in promoted:
                msg += (
                    f"  <b>{r['name']}</b>\n"
                    f"  Sharpe={r['sharpe']:+.2f} PF={r['profit_factor']:.2f} "
                    f"WR={r['win_rate']:.0f}%\n"
                )

        if watched:
            msg += "\n🟡 <b>WATCHLIST (promising but needs more testing):</b>\n"
            for r in watched:
                msg += f"  {r['name']}: Sharpe={r['sharpe']:+.2f}\n"

        if rejected:
            msg += f"\n🔴 Rejected: {len(rejected)} strategies below threshold\n"

        send_message(msg)
    except Exception:
        pass


def get_promoted_strategies() -> list[dict]:
    """Get all strategies promoted to paper trading."""
    pipeline = load_json(PIPELINE_LOG, default={"strategies": []})
    return [s for s in pipeline["strategies"] if s.get("status") == "PROMOTED"]


if __name__ == "__main__":
    process_new_papers()
