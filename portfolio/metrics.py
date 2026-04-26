"""
portfolio/metrics.py

Performance metrics computation for the portfolio simulator.
Pure functions — no state, no side effects.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from analytics.drawdown import max_drawdown, ulcer_index, worst_month


# ──────────────────────────────────────────
# CAGR
# ──────────────────────────────────────────


def compute_cagr(initial: float, final: float, years: float) -> float:
    """Compound annual growth rate (%)."""
    if years <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return round(((final / initial) ** (1 / years) - 1) * 100, 2)


# ──────────────────────────────────────────
# Sharpe
# ──────────────────────────────────────────


def compute_equity_sharpe(equity_curve: list[float], min_days: int = 30) -> tuple[float, bool]:
    """Sharpe ratio from daily equity-curve returns."""
    if len(equity_curve) < min_days + 1:
        return 0.0, False
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    if len(returns) == 0 or np.std(returns) == 0 or np.isnan(np.std(returns)):
        return 0.0, False
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    return round(sharpe, 2), True


# ──────────────────────────────────────────
# Turnover
# ──────────────────────────────────────────


def compute_turnover(
    trade_log: list,
    equity_curve: list[tuple[date, float]],
    initial_capital: float,
    dates: list,
) -> tuple[float, float]:
    """
    Compute total notional turnover and annualized turnover ratio.

    Returns
    -------
    total_turnover : float
        Sum of abs(entry_notional) + abs(exit_notional) for all trades.
    turnover_annual : float
        (total_turnover / avg_equity) / years
    """
    total_turnover_val = 0.0
    for t in trade_log:
        entry_notional = t.quantity * t.entry_price
        exit_notional = t.quantity * t.exit_price
        total_turnover_val += abs(entry_notional) + abs(exit_notional)

    avg_equity = np.mean([e for _, e in equity_curve]) if equity_curve else initial_capital
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25) if len(dates) >= 2 else 1.0
    turnover_ratio = total_turnover_val / avg_equity if avg_equity > 0 else 0.0
    turnover_annual = turnover_ratio / years if years > 0 else 0.0

    return round(total_turnover_val, 2), round(turnover_annual, 2)


# ──────────────────────────────────────────
# Trade-level metrics
# ──────────────────────────────────────────


def compute_trade_metrics(trade_log: list):
    """
    Compute win/loss statistics from a list of trade-like objects.

    Returns dict with: total_trades, win_rate, expectancy, profit_factor,
    avg_win, avg_loss.
    """
    if not trade_log:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    total = len(trade_log)
    wins = [t for t in trade_log if t.pnl > 0]
    losses = [t for t in trade_log if t.pnl <= 0]

    win_rate = round(len(wins) / total * 100, 1)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0

    expectancy = round(
        (len(wins) / total) * avg_win + (len(losses) / total) * avg_loss, 2
    )

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = round(gross_profit / gross_loss, 2)

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
    }


# ──────────────────────────────────────────
# Summary print
# ──────────────────────────────────────────


def print_portfolio_summary(
    cagr: float,
    max_dd: float,
    sharpe: float,
    sharpe_valid: bool,
    profit_factor: float,
    ulcer: float,
    turnover_annual: float,
    total_trades: int,
    worst_month_info: dict,
) -> None:
    """Print compact one-line portfolio summary."""
    sharpe_str = f"{sharpe:.2f}" if sharpe_valid else "N/A"
    pf_str = "INF" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    turnover_str = (
        f"{turnover_annual:.1f}x/yr"
        if turnover_annual < 100
        else f"{turnover_annual:.0f}x/yr"
    )
    print(
        f"Portfolio: CAGR={cagr:.1f}% | MaxDD={max_dd:.1f}% "
        f"| Sharpe={sharpe_str} | PF={pf_str} | Ulcer={ulcer:.2f} "
        f"| Turnover={turnover_str}"
    )
    if worst_month_info:
        print(
            f"  Worst Month: {worst_month_info['month']} "
            f"({worst_month_info['return_pct']:.1f}%)"
        )
    if turnover_annual > 3.0 and total_trades > 5:
        print(
            f"  ⚠️  High turnover ({turnover_annual:.1f}x/yr) "
            f"increases transaction costs"
        )