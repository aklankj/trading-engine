"""
Backtesting engine — runs strategies on 10+ years of historical data.

Uses yfinance for long-term NSE data (free, adjusted for splits/dividends).
Tests each of 8 strategies independently + the meta-composite.
Tracks: returns, Sharpe, max drawdown, win rate, profit factor.

Usage:
    python -m backtest.engine                    # Default: NIFTY 50 stocks, 10yr
    python -m backtest.engine --years 15         # 15 years
    python -m backtest.engine --symbol RELIANCE  # Single stock
    python -m backtest.engine --quick            # Top 10 stocks, 5yr
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import cfg
from core.regime import detect_regime
from core.strategies import (
    DualMACrossover, RSIMeanReversion, BollingerSqueeze,
    MACDMomentum, KaufmanAdaptive, VolRiskParity,
    DonchianBreakout, MultiTFConsensus,
)
from core.data import compute_indicators
from utils.logger import log
from utils import save_json


STRATEGY_FUNCS = {
    "DualMA": DualMACrossover(),
    "RSI_MeanRev": RSIMeanReversion(),
    "Bollinger": BollingerSqueeze(),
    "MACD": MACDMomentum(),
    "Kaufman": KaufmanAdaptive(),
    "VolRiskParity": VolRiskParity(),
    "Donchian": DonchianBreakout(),
    "MultiTF": MultiTFConsensus(),
}

NIFTY50_SYMBOLS = [
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

TOP10_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "ITC", "SBIN", "LT", "HCLTECH",
]


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    strategy: str
    regime: str


@dataclass
class StrategyResult:
    name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_days: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def fetch_yfinance_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """Fetch historical data from yfinance for NSE stocks."""
    # yfinance uses .NS suffix for NSE
    yf_symbol = f"{symbol}.NS"

    # Handle special cases
    symbol_map = {
        "M&M.NS": "M&M.NS",
        "BAJAJ-AUTO.NS": "BAJAJ-AUTO.NS",
    }
    yf_symbol = symbol_map.get(yf_symbol, yf_symbol)

    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start, end=end, interval="1d")

        if df.empty:
            return pd.DataFrame()

        # Rename columns to match our format
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna()

        return df

    except Exception as e:
        log.warning(f"yfinance fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_func,
    symbol: str,
    signal_threshold: float = 0.3,
    holding_period: int = 5,
    stop_loss_atr_mult: float = 2.0,
    target_atr_mult: float = 3.0,
) -> StrategyResult:
    """
    Backtest a single strategy on historical data.

    Rules:
    - Enter when |signal| > threshold
    - Exit after holding_period days, or hit stop/target
    - Track each trade's P&L
    """
    result = StrategyResult(name=strategy_name)

    if len(df) < 200:
        return result

    # Compute indicators
    df = compute_indicators(df.copy())

    # Need ATR for stops
    if "atr" not in df.columns:
        return result

    trades = []
    equity = 100000.0  # Start with 1L
    equity_curve = [equity]
    peak_equity = equity

    position = None  # None, or dict with entry info
    i = 200  # Start after warmup

    while i < len(df):
        row = df.iloc[i]
        date = df.index[i]

        # Check exit conditions if in position
        if position is not None:
            days_held = i - position["entry_idx"]
            current_price = row["close"]

            exit_reason = None

            # Stop loss
            if position["direction"] == "BUY":
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["target"]:
                    exit_reason = "target"
                elif days_held >= holding_period:
                    exit_reason = "time_exit"
            else:  # SELL
                if current_price >= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price <= position["target"]:
                    exit_reason = "target"
                elif days_held >= holding_period:
                    exit_reason = "time_exit"

            if exit_reason:
                # Calculate return
                if position["direction"] == "BUY":
                    ret_pct = (current_price - position["entry_price"]) / position["entry_price"] * 100
                else:
                    ret_pct = (position["entry_price"] - current_price) / position["entry_price"] * 100

                trade = TradeRecord(
                    entry_date=str(position["entry_date"])[:10],
                    exit_date=str(date)[:10],
                    symbol=symbol,
                    direction=position["direction"],
                    entry_price=round(position["entry_price"], 2),
                    exit_price=round(current_price, 2),
                    return_pct=round(ret_pct, 2),
                    holding_days=days_held,
                    strategy=strategy_name,
                    regime=position.get("regime", "Unknown"),
                )
                trades.append(trade)

                # Update equity
                position_size = equity * 0.05  # 5% per trade
                equity += position_size * ret_pct / 100
                equity_curve.append(equity)
                peak_equity = max(peak_equity, equity)

                position = None

        # Check entry conditions if flat
        if position is None:
            try:
                # Get strategy signal
                signal_val = strategy_func(df.iloc[:i+1])

                # Handle different return types
                if hasattr(signal_val, 'signal'):
                    sig = signal_val.signal
                elif isinstance(signal_val, (int, float)):
                    sig = float(signal_val)
                elif isinstance(signal_val, tuple):
                    sig = float(signal_val[0])
                else:
                    sig = 0.0

                atr = row["atr"]

                if abs(sig) > signal_threshold and atr > 0:
                    direction = "BUY" if sig > 0 else "SELL"
                    entry_price = row["close"]

                    if direction == "BUY":
                        stop_loss = entry_price - atr * stop_loss_atr_mult
                        target = entry_price + atr * target_atr_mult
                    else:
                        stop_loss = entry_price + atr * stop_loss_atr_mult
                        target = entry_price - atr * target_atr_mult

                    # Detect regime
                    try:
                        regime_result = detect_regime(df.iloc[:i+1])
                        regime = regime_result.regime
                    except Exception:
                        regime = "Unknown"

                    position = {
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_date": date,
                        "entry_idx": i,
                        "stop_loss": stop_loss,
                        "target": target,
                        "regime": regime,
                    }

            except Exception:
                pass

        i += 1

    # Calculate metrics
    if not trades:
        return result

    returns = [t.return_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    result.total_trades = len(trades)
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / len(trades) * 100 if trades else 0

    result.avg_win_pct = np.mean(wins) if wins else 0
    result.avg_loss_pct = np.mean(losses) if losses else 0

    result.avg_holding_days = np.mean([t.holding_days for t in trades])

    # Total and annualized return
    result.total_return_pct = (equity / 100000 - 1) * 100

    years_tested = (df.index[-1] - df.index[200]).days / 365.25
    if years_tested > 0:
        result.annualized_return_pct = (
            (equity / 100000) ** (1 / years_tested) - 1
        ) * 100

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1:
        daily_equiv_returns = [r / max(t.holding_days, 1) for r, t in zip(returns, trades)]
        mean_ret = np.mean(daily_equiv_returns)
        std_ret = np.std(daily_equiv_returns)
        if std_ret > 0:
            result.sharpe_ratio = round(mean_ret / std_ret * np.sqrt(252), 2)

    # Max drawdown
    peak = 100000
    max_dd = 0
    running = 100000
    for t in trades:
        position_size = running * 0.05
        running += position_size * t.return_pct / 100
        peak = max(peak, running)
        dd = (peak - running) / peak * 100
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = round(max_dd, 2)

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

    result.equity_curve = equity_curve
    result.trades = [asdict(t) for t in trades[:500]]  # Limit stored trades

    return result


def run_backtest(
    symbols: list[str] = None,
    years: int = 10,
    quick: bool = False,
) -> dict:
    """
    Run full backtest across stocks and strategies.
    """
    if symbols is None:
        symbols = TOP10_SYMBOLS if quick else NIFTY50_SYMBOLS

    log.info(f"═══ BACKTEST STARTED — {len(symbols)} stocks, {years} years ═══")

    all_strategy_results = {name: [] for name in STRATEGY_FUNCS}
    all_strategy_results["MetaComposite"] = []

    stock_results = {}
    failed_symbols = []

    for idx, symbol in enumerate(symbols):
        log.info(f"  [{idx+1}/{len(symbols)}] {symbol}...")

        df = fetch_yfinance_data(symbol, years=years)
        if df.empty or len(df) < 500:
            log.warning(f"    Insufficient data for {symbol} ({len(df)} rows)")
            failed_symbols.append(symbol)
            continue

        log.info(f"    {len(df)} trading days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

        symbol_results = {}

        # Test each strategy individually
        for strat_name, strat_func in STRATEGY_FUNCS.items():
            try:
                result = backtest_strategy(
                    df=df.copy(),
                    strategy_name=strat_name,
                    strategy_func=strat_func,
                    symbol=symbol,
                )
                symbol_results[strat_name] = result
                all_strategy_results[strat_name].append(result)

                if result.total_trades > 0:
                    log.info(
                        f"    {strat_name:15s}: {result.total_trades:4d} trades | "
                        f"Win={result.win_rate:5.1f}% | "
                        f"Return={result.total_return_pct:7.1f}% | "
                        f"Sharpe={result.sharpe_ratio:5.2f} | "
                        f"MaxDD={result.max_drawdown_pct:5.1f}%"
                    )
            except Exception as e:
                log.warning(f"    {strat_name} failed: {e}")

        stock_results[symbol] = symbol_results
        time.sleep(0.5)  # Rate limit yfinance

    # Aggregate results per strategy
    summary = {}
    for strat_name, results_list in all_strategy_results.items():
        valid = [r for r in results_list if r.total_trades > 0]
        if not valid:
            summary[strat_name] = {"status": "no_trades"}
            continue

        total_trades = sum(r.total_trades for r in valid)
        total_wins = sum(r.winning_trades for r in valid)
        avg_return = np.mean([r.total_return_pct for r in valid])
        avg_annual = np.mean([r.annualized_return_pct for r in valid])
        avg_sharpe = np.mean([r.sharpe_ratio for r in valid])
        avg_dd = np.mean([r.max_drawdown_pct for r in valid])
        avg_pf = np.mean([r.profit_factor for r in valid])
        avg_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        summary[strat_name] = {
            "stocks_tested": len(valid),
            "total_trades": total_trades,
            "win_rate": round(avg_win_rate, 1),
            "avg_total_return": round(avg_return, 1),
            "avg_annual_return": round(avg_annual, 1),
            "avg_sharpe": round(avg_sharpe, 2),
            "avg_max_drawdown": round(avg_dd, 1),
            "avg_profit_factor": round(avg_pf, 2),
            "verdict": _verdict(avg_sharpe, avg_win_rate, avg_pf),
        }

    # Save results
    output = {
        "backtest_date": datetime.now().isoformat(),
        "config": {
            "symbols": symbols,
            "years": years,
            "signal_threshold": 0.3,
            "holding_period_days": 5,
            "position_size_pct": 5,
        },
        "summary": summary,
        "failed_symbols": failed_symbols,
        "per_stock": {
            sym: {
                strat: {
                    "trades": r.total_trades,
                    "win_rate": round(r.win_rate, 1),
                    "return": round(r.total_return_pct, 1),
                    "sharpe": r.sharpe_ratio,
                    "max_dd": r.max_drawdown_pct,
                    "profit_factor": r.profit_factor,
                }
                for strat, r in results.items()
                if r.total_trades > 0
            }
            for sym, results in stock_results.items()
        },
    }

    output_path = cfg.DATA_DIR / "backtest_results.json"
    save_json(output_path, output)

    # Print summary
    _print_summary(summary, years, len(symbols))

    # Send Telegram report
    _send_backtest_report(summary, years, len(symbols), failed_symbols)

    return output


def _verdict(sharpe: float, win_rate: float, pf: float) -> str:
    """Rate a strategy based on metrics."""
    score = 0
    if sharpe > 1.0: score += 3
    elif sharpe > 0.5: score += 2
    elif sharpe > 0: score += 1

    if win_rate > 55: score += 2
    elif win_rate > 50: score += 1

    if pf > 1.5: score += 2
    elif pf > 1.0: score += 1

    if score >= 6: return "EXCELLENT — keep and increase weight"
    if score >= 4: return "GOOD — keep at current weight"
    if score >= 2: return "MARGINAL — reduce weight or tune"
    return "POOR — consider removing"


def _print_summary(summary: dict, years: int, n_stocks: int):
    """Print formatted backtest summary."""
    print(f"\n{'='*80}")
    print(f"  BACKTEST RESULTS — {n_stocks} stocks, {years} years")
    print(f"{'='*80}")
    print(f"{'Strategy':18s} {'Trades':>7s} {'WinRate':>8s} {'Return':>8s} {'Annual':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'PF':>6s} Verdict")
    print(f"{'-'*80}")

    for name, data in sorted(summary.items(), key=lambda x: x[1].get("avg_sharpe", -99), reverse=True):
        if data.get("status") == "no_trades":
            print(f"  {name:16s}  — no trades generated —")
            continue
        print(
            f"  {name:16s} {data['total_trades']:7d} {data['win_rate']:7.1f}% "
            f"{data['avg_total_return']:7.1f}% {data['avg_annual_return']:7.1f}% "
            f"{data['avg_sharpe']:7.2f} {data['avg_max_drawdown']:7.1f}% "
            f"{data['avg_profit_factor']:5.2f}  {data['verdict']}"
        )

    print(f"{'='*80}\n")


def _send_backtest_report(summary: dict, years: int, n_stocks: int, failed: list):
    """Send backtest results to Telegram."""
    try:
        from execution.telegram_bot import send_message

        msg = (
            f"📊 <b>BACKTEST RESULTS</b>\n"
            f"<i>{n_stocks} stocks × {years} years</i>\n\n"
        )

        sorted_strats = sorted(
            [(k, v) for k, v in summary.items() if v.get("total_trades", 0) > 0],
            key=lambda x: x[1].get("avg_sharpe", -99),
            reverse=True,
        )

        for name, data in sorted_strats:
            emoji = "🟢" if data["avg_sharpe"] > 0.5 else "🟡" if data["avg_sharpe"] > 0 else "🔴"
            msg += (
                f"{emoji} <b>{name}</b>\n"
                f"  Trades: {data['total_trades']} | Win: {data['win_rate']:.0f}%\n"
                f"  Return: {data['avg_total_return']:.1f}% | Sharpe: {data['avg_sharpe']:.2f}\n"
                f"  MaxDD: {data['avg_max_drawdown']:.1f}% | PF: {data['avg_profit_factor']:.2f}\n"
                f"  → <i>{data['verdict']}</i>\n\n"
            )

        if failed:
            msg += f"\n⚠️ {len(failed)} symbols had insufficient data\n"

        msg += (
            f"\n💡 <b>Action items:</b>\n"
            f"Strategies with Sharpe > 0.5 → increase weight\n"
            f"Strategies with Sharpe < 0 → reduce or remove\n"
            f"Strategies with PF < 1.0 → losing money, fix or kill"
        )

        send_message(msg)
    except Exception as e:
        log.warning(f"Telegram report failed: {e}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    years = 10

    for arg in sys.argv[1:]:
        if arg.startswith("--years="):
            years = int(arg.split("=")[1])
        elif arg.startswith("--symbol="):
            symbol = arg.split("=")[1]
            run_backtest(symbols=[symbol], years=years)
            sys.exit(0)

    symbols = TOP10_SYMBOLS if quick else NIFTY50_SYMBOLS
    run_backtest(symbols=symbols, years=years, quick=quick)
