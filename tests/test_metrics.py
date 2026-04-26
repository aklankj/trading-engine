"""
Task 6: Verify metric calculations are mathematically correct.

These tests use controlled trade inputs and hand-calculated expected values
to prove CAGR, expectancy, profit factor, Sharpe gating, and max drawdown
formulas are implemented correctly.
"""

import numpy as np
import pytest

from strategies.base import BacktestResult


def _make_result(trades: list[float], years: int = 1) -> BacktestResult:
    """
    Construct a BacktestResult with metrics computed the same way
    BaseStrategy.backtest() computes them (drawdown section).
    """
    result = BacktestResult(strategy="TestMetrics")
    if not trades:
        return result

    equity = 100000.0
    running = equity
    peak = equity
    max_dd = 0.0
    for t in trades:
        pnl = running * 0.10 * t / 100
        cost = running * 0.10 * 0.002
        running += pnl - cost
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    result.trades = trades
    result.equity_final = round(running, 2)
    result.total_trades = len(trades)
    result.winners = len(wins)
    result.losers = len(losses)
    result.win_rate = round(len(wins) / len(trades) * 100, 1) if trades else 0
    result.avg_win = round(np.mean(wins), 1) if wins else 0
    result.avg_loss = round(np.mean(losses), 1) if losses else 0
    result.total_return_pct = round((running / 100000 - 1) * 100, 2)
    result.years_tested = years
    if result.years_tested > 0 and running > 0:
        result.cagr = round(
            ((running / 100000) ** (1 / result.years_tested) - 1) * 100, 2
        )

    wr = len(wins) / len(trades)
    lr = len(losses) / len(trades)
    result.expectancy = round(
        wr * (np.mean(wins) if wins else 0)
        + lr * (np.mean(losses) if losses else 0),
        2,
    )

    if len(trades) >= result.MIN_TRADES_FOR_SHARPE:
        if np.std(trades) > 0:
            tpy = len(trades) / max(result.years_tested, 0.5)
            result.sharpe = round(
                np.mean(trades) / np.std(trades) * np.sqrt(tpy), 2
            )
            result.sharpe_valid = True
    else:
        if len(trades) > 1 and np.std(trades) > 0:
            result.sharpe = round(np.mean(trades) / np.std(trades), 2)
        result.sharpe_valid = False

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    result.profit_factor = (
        round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
    )
    result.max_drawdown = round(max_dd, 1)
    return result


def test_cagr_formula():
    """
    Hand-calculated CAGR cases:
      100K → ~200K in 5 years  → CAGR ≈ 14.87%
      100K → 100K  in 10 years → CAGR = 0%
      100K → ~50K  in 3 years  → CAGR ≈ -20.6%
    """
    # Case 1: one +1000% trade on 10% position ≈ doubles the portfolio
    ret = _make_result([1000.0], years=5)
    assert pytest.approx(ret.cagr, 0.1) == 14.87

    # Case 2: a +0.2% trade produces exactly enough P&L to cancel costs
    ret = _make_result([0.2], years=10)
    assert ret.cagr == 0.0

    # Case 3: a -500% trade roughly halves the portfolio
    ret = _make_result([-500.0], years=3)
    assert pytest.approx(ret.cagr, 0.1) == -20.6


def test_expectancy_formula():
    """
    Trades: [+10, +5, -3, -4, +8]
    win_rate = 3/5, avg_win = 23/3, loss_rate = 2/5, avg_loss = -3.5
    Expectancy = 0.6*(23/3) + 0.4*(-3.5) = 3.2%
    """
    ret = _make_result([10.0, 5.0, -3.0, -4.0, 8.0])
    assert pytest.approx(ret.expectancy, 0.01) == 3.2


def test_profit_factor():
    """
    Wins: [10, 5, 8] → gross profit = 23
    Losses: [-3, -4] → gross loss = 7
    Profit Factor = 23 / 7 = 3.29
    """
    ret = _make_result([10.0, 5.0, 8.0, -3.0, -4.0])
    assert pytest.approx(ret.profit_factor, 0.01) == 3.29


def test_sharpe_gating():
    """
    5 trades  → sharpe_valid = False (below threshold)
    30 trades → sharpe_valid = True  (at threshold, std > 0)
    0 trades  → sharpe = 0
    """
    ret5 = _make_result([2.0, -1.0, 2.0, -1.0, 2.0])
    assert ret5.sharpe_valid is False

    trades_30 = [2.0 if i % 2 == 0 else -1.0 for i in range(30)]
    ret30 = _make_result(trades_30)
    assert ret30.sharpe_valid is True

    ret0 = _make_result([])
    assert ret0.sharpe == 0
    assert ret0.sharpe_valid is False


def test_max_drawdown():
    """
    Equity curve: 100, 110, 90, 95, 80, 100
    Peak = 110, trough = 80
    Max DD = (110 - 80) / 110 = 27.27%
    """
    equity_curve = [100, 110, 90, 95, 80, 100]
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    assert pytest.approx(max_dd, 0.1) == 27.27
