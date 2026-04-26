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
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import CORE_STRATEGIES, print_backtest_summary
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


def run_backtest(
    symbols: list[str] = None,
    years: int = 10,
    walk_forward: bool = False,
) -> dict:
    """
    Backtest all registered strategies across stocks.
    Uses the EXACT same strategy code as live trading.
    """
    if symbols is None:
        symbols = TOP10

    log.info(f"═══ UNIFIED BACKTEST — {len(symbols)} stocks, {years}yr ═══")

    # Aggregate results per strategy
    agg: dict[str, list[BacktestResult]] = {name: [] for name in CORE_STRATEGIES}

    for idx, symbol in enumerate(symbols):
        log.info(f"  [{idx+1}/{len(symbols)}] {symbol}...")

        df = fetch_data(symbol, years=years)
        if df.empty or len(df) < 500:
            log.warning(f"    Insufficient data ({len(df)} rows)")
            continue

        if walk_forward:
            # Walk-forward: train on first 70%, test on last 30%
            split = int(len(df) * 0.7)
            test_df = df.iloc[split:]
            log.info(f"    Walk-forward: testing on {len(test_df)} bars ({df.index[split].strftime('%Y-%m-%d')} onwards)")
        else:
            test_df = df

        for strat_name, strategy in CORE_STRATEGIES.items():
            try:
                result = strategy.backtest(test_df)
                agg[strat_name].append(result)

                if result.total_trades > 0:
                    log.info(
                        f"    {strat_name:18s}: {result.total_trades:3d} trades | "
                        f"WR={result.win_rate:5.1f}% | Sharpe={result.sharpe:+5.2f} | "
                        f"PF={result.profit_factor:.2f}"
                    )
            except Exception as e:
                log.warning(f"    {strat_name} failed: {e}")

        time.sleep(0.5)

    # Aggregate per strategy across all stocks
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

    # Print results
    mode = "WALK-FORWARD" if walk_forward else "IN-SAMPLE"
    print(f"\n{'='*105}")
    print(f"  UNIFIED BACKTEST ({mode}) — {len(symbols)} stocks, {years} years")
    print(f"  Strategy code: strategies.registry (SAME as live engine)")
    print(f"{'='*105}")
    print(f"  {'Strategy':18s} {'Trades':>6s} {'WR':>6s} {'CAGR':>7s} {'Expect':>7s} {'Sharpe':>10s} {'PF':>5s} {'MaxDD':>6s} {'AvgWin':>7s} {'AvgLoss':>8s} {'Hold':>5s}")
    print(f"  {'-'*102}")

    for name, r in sorted(summary.items(), key=lambda x: x[1].cagr, reverse=True):
        if r.total_trades == 0:
            print(f"  {name:18s}  — no trades —")
            continue

        if r.sharpe_valid:
            sharpe_str = f"{r.sharpe:+6.2f}"
        elif r.total_trades > 1:
            sharpe_str = f"{r.sharpe:+5.2f}(n<30)"
        else:
            sharpe_str = "   N/A"

        verdict = "🏆" if r.cagr > 10 and r.profit_factor > 1.5 else \
                  "✅" if r.cagr > 5 and r.profit_factor > 1.2 else \
                  "🟡" if r.cagr > 0 else "❌"

        print(
            f"  {verdict} {name:16s} {r.total_trades:6d} {r.win_rate:5.1f}% "
            f"{r.cagr:+6.1f}% {r.expectancy:+6.2f}% {sharpe_str:>10s} "
            f"{r.profit_factor:4.2f} {r.max_drawdown:5.1f}% "
            f"{r.avg_win:+6.1f}% {r.avg_loss:+7.1f}% {r.avg_hold_days:4.0f}d"
        )

    print(f"{'='*105}")
    print(f"  Note: Sharpe marked (n<30) has insufficient trades for statistical validity.")
    print(f"  Primary metrics: CAGR (compound growth), Expectancy (avg return/trade), PF (profit/loss ratio)")

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
            sharpe_note = f"Sharpe={r.sharpe:+.2f}" if r.sharpe_valid else f"Sharpe={r.sharpe:+.2f}*"
            msg += (
                f"{emoji} <b>{name}</b>: "
                f"CAGR={r.cagr:+.1f}% E={r.expectancy:+.2f}% "
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
