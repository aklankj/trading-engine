"""
Unified Backtester V2.

Uses the SAME strategy code as live engine (strategies.registry).
No divergence between backtest and live is possible.

Usage:
    python -m backtest.unified                     # All strategies, 10 stocks, 10yr
    python -m backtest.unified --full              # All strategies, 47 stocks
    python -m backtest.unified --symbol=RELIANCE   # Single stock
    python -m backtest.unified --years=15          # 15 years
    python -m backtest.unified --walkforward       # Walk-forward validation
"""

import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import CORE_STRATEGIES
from strategies.base import BacktestResult
from config.settings import cfg
from utils.logger import log
from utils import save_json

TOP10 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "ITC", "SBIN", "LT", "HCLTECH",
]

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
    "ITC", "SBIN", "LT", "HCLTECH", "BAJFINANCE", "HINDUNILVR",
    "MARUTI", "KOTAKBANK", "SUNPHARMA", "WIPRO", "M&M", "ULTRACEMCO",
    "AXISBANK", "ASIANPAINT", "TITAN", "NESTLEIND", "JSWSTEEL",
    "TATASTEEL", "BAJAJ-AUTO", "TECHM", "INDUSINDBK", "CIPLA",
    "DRREDDY", "EICHERMOT", "BPCL", "COALINDIA", "DIVISLAB",
    "BRITANNIA", "HEROMOTOCO", "HINDALCO", "GRASIM", "TRENT",
    "ADANIPORTS", "PIDILITIND", "NTPC", "POWERGRID", "ONGC",
    "APOLLOHOSP", "SBILIFE", "TATACONSUM", "BAJAJFINSV",
]

# 100+ stocks: NIFTY 100 equivalent (NIFTY 50 + Next 50 + liquid midcaps)
NIFTY100_PLUS = NIFTY50 + [
    # NIFTY Next 50
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
    "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "GAIL", "GODREJCP",
    "HAVELLS", "HAL", "ICICIPRULI", "INDHOTEL", "INDUSTOWER",
    "IOC", "IRCTC", "JINDALSTEL", "LUPIN", "MARICO",
    "MAXHEALTH", "MOTHERSON", "NAUKRI", "NHPC", "OBEROIRLTY",
    "OFSS", "PIIND", "PNB", "POLYCAB", "RECLTD",
    "SBICARD", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
    "TVSMOTOR", "VEDL", "ZOMATO", "ZYDUSLIFE",
    # Liquid midcaps for diversity
    "ASTRAL", "ATUL", "BATAINDIA", "CDSL", "CUMMINSIND",
    "DEEPAKNTR", "DMART", "ESCORTS", "FEDERALBNK", "FORTIS",
    "IPCALAB", "JUBLFOOD", "LALPATHLAB", "LTIM", "MPHASIS",
    "MUTHOOTFIN", "PAGEIND", "PERSISTENT", "PETRONET", "PVRINOX",
    "SAIL", "TATAELXSI", "VOLTAS",
]


from backtest.walkforward import generate_walkforward_windows, summarize_walkforward
from backtest.reporting import (
    print_backtest_header,
    print_backtest_table,
    print_backtest_footer,
    print_compact_summary,
    print_benchmarks,
    print_walkforward_header,
    print_walkforward_table,
    print_walkforward_summary,
)


def _run_window(
    data: dict[str, pd.DataFrame],
    window: dict,
    test_capital: float = 100_000,
) -> dict | None:
    """
    Run portfolio simulator on a single walk-forward test window.

    Returns dict with window metrics or None on failure.
    """
    try:
        from portfolio.simulator import simulate

        result = simulate(
            data=data,
            start_date=window["test_start"],
            end_date=window["test_end"],
            initial_capital=test_capital,
        )

        if result.total_trades == 0:
            return None

        return {
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "cagr": result.cagr,
            "max_dd": result.max_drawdown,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
        }
    except Exception as e:
        log.warning(f"    Window failed: {e}")
        return None


# ──────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────


def fetch_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """Fetch historical data from yfinance."""
    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start, end=end, interval="1d")

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]].dropna()

    except Exception as e:
        return pd.DataFrame()


# ──────────────────────────────────────────
# Main backtest
# ──────────────────────────────────────────


def run_backtest(
    symbols: list[str] = None,
    years: int = 10,
    walk_forward: bool = False,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> dict:
    """
    Backtest all registered strategies across stocks.
    Uses the EXACT same strategy code as live trading.

    When walk_forward=True, runs rolling out-of-sample windows using
    the portfolio simulator (portfolio.simulator.simulate).
    Each window: train_years reserved context, test_years OOS evaluation,
    rolling forward by step_years.

    NOTE: Train windows are currently reserved periods for future
    parameter calibration / optimization. Since strategies are
    rule-based and not fitted, no in-sample fitting occurs yet.
    """
    if symbols is None:
        symbols = TOP10

    log.info(f"═══ UNIFIED BACKTEST — {len(symbols)} stocks, {years}yr ═══")

    if walk_forward:
        return _run_walkforward_backtest(
            symbols, years, train_years, test_years, step_years
        )

    # ── Standard backtest (non walk-forward) ──
    # Aggregate results per strategy
    agg: dict[str, list[BacktestResult]] = {name: [] for name in CORE_STRATEGIES}
    data: dict[str, pd.DataFrame] = {}

    for idx, symbol in enumerate(symbols):
        log.info(f"  [{idx+1}/{len(symbols)}] {symbol}...")

        df = fetch_data(symbol, years=years)
        if df.empty or len(df) < 500:
            log.warning(f"    Insufficient data ({len(df)} rows)")
            continue
        data[symbol] = df

        for strat_name, strategy in CORE_STRATEGIES.items():
            try:
                result = strategy.backtest(df)
                agg[strat_name].append(result)

                if result.total_trades > 0:
                    log.info(
                        f"    {strat_name:18s}: {result.total_trades:3d} trades | "
                        f"WR={result.win_rate:5.1f}% | Sharpe={result.sharpe:5.2f} | "
                        f"PF={result.profit_factor:.2f}"
                    )
            except Exception as e:
                log.warning(f"    {strat_name} failed: {e}")

        if cfg.DEBUG_SLEEP:
            time.sleep(0.5)

    # Aggregate per strategy across all stocks
    summary = _aggregate_results(agg)

    # Print results
    mode = "IN-SAMPLE"
    print_backtest_header(mode, len(symbols), years)
    print_backtest_table(summary)
    print_backtest_footer()
    print_compact_summary(summary)

    # ── Benchmarks ──
    if data:
        from analytics.benchmark import run_nifty_proxy, run_equal_weight, run_buy_and_hold

        all_dates_set: set[pd.Timestamp] = set()
        for sym in symbols:
            if sym in data:
                all_dates_set.update(data[sym].index)
        if all_dates_set:
            sorted_dates = sorted(all_dates_set)
            start_date = sorted_dates[0]
            end_date = sorted_dates[-1]

            b_nifty = run_nifty_proxy(data, start_date=start_date, end_date=end_date)
            b_eqw = run_equal_weight(data, symbols, start_date=start_date, end_date=end_date)
            b_bnh = run_buy_and_hold(data, symbols, start_date=start_date, end_date=end_date)

            print_benchmarks(b_nifty, b_eqw, b_bnh)

    # Save results
    output = {
        "date": datetime.now().isoformat(),
        "mode": mode,
        "symbols": symbols,
        "years": years,
        "summary": {
            name: {
                "trades": r.total_trades, "win_rate": r.win_rate,
                "cagr": r.cagr, "expectancy": r.expectancy,
                "total_return_pct": r.total_return_pct,
                "sharpe": r.sharpe, "sharpe_valid": r.sharpe_valid,
                "profit_factor": r.profit_factor,
                "max_drawdown": r.max_drawdown, "avg_win": r.avg_win,
                "avg_loss": r.avg_loss, "avg_hold_days": r.avg_hold_days,
                "years_tested": r.years_tested,
            }
            for name, r in summary.items()
        },
    }

    output_path = cfg.DATA_DIR / "backtest_results.json"
    save_json(output_path, output)
    log.info(f"Results saved to {output_path}")

    # Telegram report
    _send_report(summary, mode, len(symbols), years)

    return summary


def _run_walkforward_backtest(
    symbols: list[str],
    years: int,
    train_years: int,
    test_years: int,
    step_years: int,
) -> dict:
    """
    Run walk-forward validation using the portfolio simulator.

    Fetches data for all symbols, generates rolling train/test windows,
    runs portfolio.simulator.simulate on each test window, and prints
    a per-window table + summary metrics.
    """
    log.info(f"  Walk-Forward: {train_years}yr train / {test_years}yr test / {step_years}yr step")

    # Fetch all data
    data: dict[str, pd.DataFrame] = {}
    for idx, symbol in enumerate(symbols):
        log.info(f"  Fetching [{idx+1}/{len(symbols)}] {symbol}...")
        df = fetch_data(symbol, years=years)
        if not df.empty and len(df) >= 500:
            data[symbol] = df
        else:
            log.warning(f"    Skipping {symbol} ({len(df)} rows)")
        if cfg.DEBUG_SLEEP:
            time.sleep(0.3)

    if not data:
        log.error("No valid data for any symbol")
        return {}

    # Build unified date index across all symbols
    all_dates: set[pd.Timestamp] = set()
    for sym in symbols:
        if sym in data:
            all_dates.update(data[sym].index)
    unified_index = sorted(all_dates)
    if len(unified_index) < 2:
        log.error("Date union too short for walk-forward")
        return {}

    data_index = pd.DatetimeIndex(unified_index)

    # Generate windows
    windows = generate_walkforward_windows(
        data_index, train_years=train_years, test_years=test_years, step_years=step_years
    )

    if not windows:
        log.error(
            f"Data span too short for {train_years}yr train + {test_years}yr test. "
            f"Need at least {train_years + test_years} years."
        )
        return {}

    log.info(f"  Generated {len(windows)} walk-forward windows")

    # Run each window
    window_results: list[dict] = []
    is_results: list[dict] = []

    for w_idx, window in enumerate(windows):
        log.info(f"  Window [{w_idx+1}/{len(windows)}]: "
                 f"train={window['train_start'].strftime('%Y-%m-%d')}→{window['train_end'].strftime('%Y-%m-%d')}, "
                 f"test={window['test_start'].strftime('%Y-%m-%d')}→{window['test_end'].strftime('%Y-%m-%d')}")

        # Run OOS window
        result = _run_window(data, window)
        if result:
            window_results.append(result)
            log.info(f"    → CAGR={result['cagr']:.1f}%  MaxDD={result['max_dd']:.1f}%  Trades={result['total_trades']}")

        # Also run IS (train window) for WFE denominator
        try:
            from portfolio.simulator import simulate
            is_result = simulate(
                data=data,
                start_date=window["train_start"],
                end_date=window["train_end"],
                initial_capital=100_000,
            )
            if is_result.total_trades > 0:
                is_results.append({
                    "cagr": is_result.cagr,
                    "max_dd": is_result.max_drawdown,
                    "total_trades": is_result.total_trades,
                })
        except Exception:
            pass

        if cfg.DEBUG_SLEEP:
            time.sleep(0.2)

    # Summarize
    wf_summary = summarize_walkforward(window_results, is_results)

    # Print walk-forward results
    print_walkforward_header(len(symbols), len(windows), train_years, test_years, step_years)
    print_walkforward_table(window_results)
    print_walkforward_summary(wf_summary)

    # Save results
    output = {
        "date": datetime.now().isoformat(),
        "mode": "WALK-FORWARD",
        "symbols": symbols,
        "years": years,
        "train_years": train_years,
        "test_years": test_years,
        "step_years": step_years,
        "windows": [
            {
                "window": i + 1,
                "train_start": w["train_start"].isoformat(),
                "train_end": w["train_end"].isoformat(),
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "cagr": w["cagr"],
                "max_dd": w["max_dd"],
                "total_trades": w["total_trades"],
            }
            for i, w in enumerate(window_results)
        ],
        "summary": wf_summary,
    }

    output_path = cfg.DATA_DIR / "backtest_results.json"
    save_json(output_path, output)
    log.info(f"Results saved to {output_path}")

    return wf_summary


def _aggregate_results(
    agg: dict[str, list[BacktestResult]],
) -> dict[str, BacktestResult]:
    """Aggregate per-strategy backtest results across symbols."""
    summary = {}
    for strat_name, results_list in agg.items():
        valid = [r for r in results_list if r.total_trades > 0]
        if not valid:
            summary[strat_name] = BacktestResult(strategy=strat_name)
            continue

        total_trades = sum(r.total_trades for r in valid)
        total_wins = sum(r.winners for r in valid)

        # Pool all individual trades for proper Sharpe calculation
        all_trades = []
        for r in valid:
            all_trades.extend(r.trades)

        avg_years = np.mean([r.years_tested for r in valid if r.years_tested > 0])

        # Sharpe from pooled trades (gated at 30)
        sharpe = 0.0
        sharpe_valid = False
        if len(all_trades) >= 30 and np.std(all_trades) > 0:
            trades_per_year = len(all_trades) / max(avg_years * len(valid), 1)
            sharpe = round(np.mean(all_trades) / np.std(all_trades) * np.sqrt(trades_per_year), 2)
            sharpe_valid = True
        elif len(all_trades) > 1 and np.std(all_trades) > 0:
            sharpe = round(np.mean(all_trades) / np.std(all_trades), 2)

        # Expectancy from pooled trades
        pool_wins = [t for t in all_trades if t > 0]
        pool_losses = [t for t in all_trades if t <= 0]
        wr = len(pool_wins) / len(all_trades) if all_trades else 0
        lr = len(pool_losses) / len(all_trades) if all_trades else 0
        expectancy = round(
            wr * (np.mean(pool_wins) if pool_wins else 0) +
            lr * (np.mean(pool_losses) if pool_losses else 0), 2
        )

        combined = BacktestResult(
            strategy=strat_name,
            total_trades=total_trades,
            winners=total_wins,
            losers=total_trades - total_wins,
            win_rate=round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            equity_final=round(np.mean([r.equity_final for r in valid]), 0),
            cagr=round(np.mean([r.cagr for r in valid]), 2),
            expectancy=expectancy,
            total_return_pct=round(np.mean([r.total_return_pct for r in valid]), 2),
            years_tested=round(avg_years, 1),
            sharpe=sharpe,
            sharpe_valid=sharpe_valid,
            profit_factor=round(np.mean([r.profit_factor for r in valid]), 2),
            max_drawdown=round(np.mean([r.max_drawdown for r in valid]), 1),
            avg_win=round(np.mean([r.avg_win for r in valid if r.avg_win != 0]), 1),
            avg_loss=round(np.mean([r.avg_loss for r in valid if r.avg_loss != 0]), 1),
            avg_hold_days=round(np.mean([r.avg_hold_days for r in valid]), 0),
        )
        summary[strat_name] = combined

    return summary


def _send_report(summary, mode, n_stocks, years):
    """Send backtest results to Telegram."""
    try:
        from execution.telegram_bot import send_message

        msg = f"📊 <b>BACKTEST ({mode})</b>\n"
        msg += f"<i>{n_stocks} stocks, {years} years</i>\n\n"

        for name, r in sorted(summary.items(), key=lambda x: x[1].cagr, reverse=True):
            if r.total_trades == 0:
                continue
            emoji = "🏆" if r.cagr > 10 and r.profit_factor > 1.5 else \
                    "✅" if r.cagr > 5 and r.profit_factor > 1.2 else \
                    "🟡" if r.cagr > 0 else "❌"
            sharpe_note = f"Sharpe={r.sharpe:.2f}" if r.sharpe_valid else f"Sharpe={r.sharpe:.2f}*"
            msg += (
                f"{emoji} <b>{name}</b>: "
                f"CAGR={r.cagr:.1f}% E={r.expectancy:.2f}% "
                f"{sharpe_note} PF={r.profit_factor:.2f} "
                f"({r.total_trades} trades, {r.win_rate:.0f}% WR)\n"
            )

        msg += "\n<i>* Sharpe has insufficient trades (n less than 30) for statistical validity</i>"
        send_message(msg)
    except Exception:
        pass


if __name__ == "__main__":
    full = "--full" in sys.argv
    n100 = "--nifty100" in sys.argv
    wf = "--walkforward" in sys.argv
    years = 10

    for arg in sys.argv[1:]:
        if arg.startswith("--years="):
            years = int(arg.split("=")[1])
        elif arg.startswith("--symbol="):
            sym = arg.split("=")[1]
            run_backtest(symbols=[sym], years=years, walk_forward=wf)
            sys.exit(0)

    if n100:
        symbols = NIFTY100_PLUS
    elif full:
        symbols = NIFTY50
    else:
        symbols = TOP10

    print(f"Universe: {len(symbols)} stocks")
    run_backtest(symbols=symbols, years=years, walk_forward=wf)
