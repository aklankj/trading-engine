"""
Strategy Registry — single source of truth.

Every strategy is registered here. Both backtester and live engine
import from here. No other strategy source exists.

Usage:
    from strategies.registry import get_active_strategies, get_strategy
    
    # Get all active strategies for live trading
    strategies = get_active_strategies()
    
    # Get specific strategy
    strategy = get_strategy("QualityDipBuy")
    
    # Backtest all strategies
    from strategies.registry import backtest_all
    results = backtest_all(df)
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from strategies.base import BaseStrategy, SwingSignal, BacktestResult
from strategies.quality_dip_buy import QualityDipBuy
from strategies.annual_momentum import AnnualMomentum
from strategies.weekly_trend import WeeklyTrend
from strategies.donchian_monthly import DonchianMonthly
from strategies.all_weather import AllWeather
from strategies.weekly_daily import WeeklyDaily


# ── Core strategies (backtested, proven) ──

CORE_STRATEGIES: dict[str, BaseStrategy] = {
    "QualityDipBuy": QualityDipBuy(),
    "AnnualMomentum": AnnualMomentum(),
    "WeeklyTrend": WeeklyTrend(),
    "DonchianMonthly": DonchianMonthly(),
    "AllWeather": AllWeather(),
    "WeeklyDaily": WeeklyDaily(),
}

# Weights based on backtested Sharpe ratios
STRATEGY_WEIGHTS = {
    "QualityDipBuy": 0.25,    # Sharpe 2.73
    "AnnualMomentum": 0.25,   # Sharpe 2.69
    "WeeklyTrend": 0.20,      # Sharpe 2.28
    "DonchianMonthly": 0.15,  # Sharpe 1.35
    "AllWeather": 0.10,       # Sharpe 0.84
    "WeeklyDaily": 0.05,      # Sharpe 0.69
}

# Pipeline-promoted strategies (loaded dynamically)
_pipeline_strategies: dict[str, BaseStrategy] = {}


def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy by name."""
    if name in CORE_STRATEGIES:
        return CORE_STRATEGIES[name]
    if name in _pipeline_strategies:
        return _pipeline_strategies[name]
    raise KeyError(f"Strategy '{name}' not found")


def get_active_strategies() -> dict[str, BaseStrategy]:
    """Get all strategies active for live trading."""
    active = dict(CORE_STRATEGIES)
    active.update(_pipeline_strategies)
    return active


def get_all_signals(df) -> list[SwingSignal]:
    """Run all active strategies, return non-zero signals."""
    signals = []
    for name, strategy in get_active_strategies().items():
        try:
            sig = strategy.signal(df)
            if abs(sig.signal) > 0.1:
                signals.append(sig)
        except Exception:
            pass
    return signals


def get_composite_signal(df) -> SwingSignal:
    """
    Weighted composite of all strategy signals.
    Weights based on backtested Sharpe ratios.
    """
    signals = get_all_signals(df)
    if not signals:
        return SwingSignal(0, "HOLD", "Composite")

    weighted_sum = 0
    total_weight = 0
    best_signal = None
    best_strength = 0

    for sig in signals:
        w = STRATEGY_WEIGHTS.get(sig.strategy, 0.05)
        weighted_sum += sig.signal * w
        total_weight += w

        if abs(sig.signal) > best_strength:
            best_strength = abs(sig.signal)
            best_signal = sig

    if total_weight == 0:
        return SwingSignal(0, "HOLD", "Composite")

    composite = weighted_sum / total_weight

    if best_signal and abs(composite) > 0.3:
        direction = "BUY" if composite > 0 else "SELL"
        agreeing = sum(1 for s in signals if (s.signal > 0) == (composite > 0))

        # Use best signal's SL/TGT — but NEVER zero
        sl = best_signal.stop_loss
        tgt = best_signal.target

        if sl <= 0 and best_signal.direction == "BUY":
            price = df["close"].iloc[-1]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            sl = price - 2.5 * atr
            tgt = price + 4 * atr
        elif sl <= 0 and best_signal.direction == "SELL":
            price = df["close"].iloc[-1]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            sl = price + 2.5 * atr
            tgt = price - 4 * atr

        return SwingSignal(
            signal=composite,
            direction=direction,
            strategy="Composite",
            stop_loss=sl,
            target=tgt,
            hold_days=best_signal.hold_days,
            confidence=agreeing / len(signals),
            reason=f"{agreeing}/{len(signals)} agree | Best: {best_signal.strategy} ({best_signal.reason})",
        )

    return SwingSignal(0, "HOLD", "Composite")


def backtest_all(df, label: str = "") -> dict[str, BacktestResult]:
    """Backtest all strategies on one DataFrame."""
    results = {}
    for name, strategy in CORE_STRATEGIES.items():
        try:
            result = strategy.backtest(df)
            results[name] = result
        except Exception as e:
            results[name] = BacktestResult(strategy=name)
    return results


def print_backtest_summary(results: dict[str, BacktestResult], symbol: str = ""):
    """Print formatted backtest comparison with proper metrics."""
    header = f"BACKTEST: {symbol}" if symbol else "BACKTEST SUMMARY"
    print(f"\n{'='*100}")
    print(f"  {header}")
    print(f"{'='*100}")
    print(f"  {'Strategy':18s} {'Trades':>6s} {'WR':>6s} {'CAGR':>7s} {'Expect':>7s} {'Sharpe':>10s} {'PF':>5s} {'MaxDD':>6s} {'AvgWin':>7s} {'AvgLoss':>8s} {'Hold':>5s}")
    print(f"  {'-'*97}")

    for name, r in sorted(results.items(), key=lambda x: x[1].cagr, reverse=True):
        if r.total_trades == 0:
            print(f"  {name:18s}  — no trades —")
            continue

        # Show Sharpe with validity marker
        if r.sharpe_valid:
            sharpe_str = f"{r.sharpe:6.2f}"
        elif r.total_trades > 1:
            sharpe_str = f"{r.sharpe:5.2f}(n<30)"
        else:
            sharpe_str = "   N/A"

        verdict = "🏆" if r.cagr > 10 and r.profit_factor > 1.5 else \
                  "✅" if r.cagr > 5 and r.profit_factor > 1.2 else \
                  "🟡" if r.cagr > 0 else "❌"

        print(
            f"  {verdict} {name:16s} {r.total_trades:6d} {r.win_rate:5.1f}% "
            f"{r.cagr:6.1f}% {r.expectancy:6.2f}% {sharpe_str:>10s} "
            f"{r.profit_factor:4.2f} {r.max_drawdown:5.1f}% "
            f"{r.avg_win:6.1f}% {r.avg_loss:7.1f}% {r.avg_hold_days:4.0f}d"
        )

    print(f"{'='*100}")
    print(f"  Note: Sharpe marked (n<30) has insufficient trades for statistical validity.")
    print(f"  Primary metrics: CAGR (compound growth), Expectancy (avg return/trade), PF (profit/loss ratio)")
