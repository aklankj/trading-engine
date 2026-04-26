"""
portfolio/simulator.py

Multi-strategy portfolio simulator with shared capital.

Runs all strategies together on a shared capital pool, enforcing
max positions, no negative cash, one position per symbol,
and transaction costs.

Re-exports all public symbols from portfolio sub-modules for backward
compatibility.  Importing from portfolio.simulator still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd

from analytics.drawdown import max_drawdown, ulcer_index, worst_month
from config.settings import cfg as _cfg
from strategies.base import BaseStrategy, Position
from strategies.registry import CORE_STRATEGIES, STRATEGY_WEIGHTS
from utils.logger import log

from portfolio.execution import fill_entry, fill_exit, compute_trade_pnl
from utils.costs import transaction_cost
from portfolio.risk import (
    compute_sector_exposure,
    filter_candidates_by_sector,
    check_cash_constraint,
)
from portfolio.metrics import (
    compute_cagr,
    compute_equity_sharpe,
    compute_turnover,
    compute_trade_metrics,
    print_portfolio_summary,
)


# ──────────────────────────────────────────
# Data structures (kept in simulator.py)
# ──────────────────────────────────────────


@dataclass
class SimPosition:
    """Internal position tracker for the simulator."""

    symbol: str
    strategy: str
    direction: str
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    entry_idx: int
    stop_loss: float
    target: float
    entry_atr: float
    highest_since_entry: float
    lowest_since_entry: float


@dataclass
class SimTrade:
    """Closed trade record."""

    symbol: str
    strategy: str
    direction: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int


@dataclass
class SimResult:
    """Portfolio-level simulation result."""

    equity_curve: list[tuple[date, float]]
    trade_log: list[SimTrade]
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sharpe_valid: bool = False
    expectancy: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    cash_utilization_avg: float = 0.0
    final_equity: float = 0.0
    initial_capital: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    ulcer_index: float = 0.0
    worst_month_return: float = 0.0
    worst_month_label: str = ""
    turnover: float = 0.0
    turnover_annual: float = 0.0
    total_slippage_cost: float = 0.0


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def _parse_date(d: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Normalize date input to Timestamp."""
    if d is None:
        return None
    return pd.to_datetime(d)


def _default_sl_tgt(
    direction: str, price: float, atr: float
) -> tuple[float, float]:
    """Default stop-loss and target when strategy returns zero values."""
    if direction == "BUY":
        sl = price - 2.5 * atr
        tgt = price + 4.0 * atr
    else:
        sl = price + 2.5 * atr
        tgt = price - 4.0 * atr
    return sl, tgt


# ──────────────────────────────────────────
# Main simulator
# ──────────────────────────────────────────


def simulate(
    data: dict[str, pd.DataFrame],
    strategies: dict[str, BaseStrategy] | None = None,
    strategy_weights: dict[str, float] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    symbols: list[str] | None = None,
    initial_capital: float = 100_000,
    max_positions: int = 10,
    position_size_pct: float = 0.05,
    allow_multiple_per_symbol: bool = False,
    slippage_pct: float = 0.0,
    sector_map: dict[str, str] | None = None,
) -> SimResult:
    """
    Simulate a portfolio of strategies with shared capital.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Pre-loaded OHLCV data per symbol.  Each DataFrame must have a
        DatetimeIndex and columns: open, high, low, close, volume.
    strategies : dict[str, BaseStrategy] | None
        Strategy instances to run.  Defaults to CORE_STRATEGIES.
    strategy_weights : dict[str, float] | None
        Weight per strategy for signal ranking.  Defaults to STRATEGY_WEIGHTS.
    start_date / end_date : optional
        Bound the simulation period.
    symbols : list[str] | None
        Subset of symbols to trade.  Defaults to all keys in *data*.
    initial_capital : float
        Starting cash.
    max_positions : int
        Hard cap on open positions.
    position_size_pct : float
        Fraction of current equity allocated per new position.
    allow_multiple_per_symbol : bool
        If False (default), only one open position per symbol at a time.
    slippage_pct : float
        Slippage applied per trade as fraction of price.
    sector_map : dict[str, str] | None
        Optional mapping of symbol -> sector name for sector exposure cap.

    Returns
    -------
    SimResult
    """
    # ── defaults ────────────────────────────
    if strategies is None:
        strategies = dict(CORE_STRATEGIES)
    if strategy_weights is None:
        strategy_weights = dict(STRATEGY_WEIGHTS)
    if symbols is None:
        symbols = list(data.keys())
    else:
        symbols = [s for s in symbols if s in data]

    if not symbols or not strategies:
        return SimResult(equity_curve=[], trade_log=[])

    sector_map = sector_map or {}
    max_sector_pct = _cfg.MAX_SECTOR_EXPOSURE_PCT

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    # ── pre-compute indicators ──────────────
    indicators: dict[tuple[str, str], pd.DataFrame] = {}
    for strat_name, strategy in strategies.items():
        for sym in symbols:
            indicators[(strat_name, sym)] = strategy.compute_indicators(data[sym].copy())

    # ── build date union ────────────────────
    all_dates: set[pd.Timestamp] = set()
    for sym in symbols:
        all_dates.update(data[sym].index)
    dates = sorted(all_dates)

    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        return SimResult(equity_curve=[], trade_log=[])

    # ── state ───────────────────────────────
    cash = float(initial_capital)
    open_positions: dict[str, SimPosition] = {}
    trade_log: list[SimTrade] = []
    equity_curve: list[tuple[date, float]] = []
    daily_deployed: list[float] = []
    total_slippage_acc = 0.0  # FIX: accumulates everywhere

    # ── daily loop ──────────────────────────
    for current_date in dates:
        # 1. Process exits
        for sym, pos in list(open_positions.items()):
            if current_date not in data[sym].index:
                continue

            df_full = indicators[(pos.strategy, sym)]
            mask = df_full.index <= current_date
            df_slice = df_full[mask]
            if len(df_slice) == 0:
                continue

            i = len(df_slice) - 1
            close_price = float(df_slice["close"].iloc[i])

            # Update watermarks
            pos.highest_since_entry = max(pos.highest_since_entry, close_price)
            pos.lowest_since_entry = min(pos.lowest_since_entry, close_price)

            base_pos = Position(
                direction=pos.direction,
                entry_price=pos.entry_price,
                entry_idx=pos.entry_idx,
                stop_loss=pos.stop_loss,
                target=pos.target,
                entry_atr=pos.entry_atr,
                highest_since_entry=pos.highest_since_entry,
                lowest_since_entry=pos.lowest_since_entry,
            )

            strategy = strategies[pos.strategy]
            should_exit_flag, exit_reason = strategy.should_exit(df_slice, i, base_pos)

            if should_exit_flag:
                exit_slip, exit_txn, exit_slip_cost = fill_exit(
                    close_price, pos.quantity, slippage_pct, pos.direction
                )
                cash += pos.quantity * exit_slip - exit_txn
                total_slippage_acc += exit_slip_cost

                pnl, pnl_pct = compute_trade_pnl(
                    exit_slip, pos.entry_price, pos.quantity,
                    pos.direction, exit_txn
                )

                trade_log.append(
                    SimTrade(
                        symbol=pos.symbol,
                        strategy=pos.strategy,
                        direction=pos.direction,
                        entry_date=pos.entry_date.date(),
                        exit_date=current_date.date(),
                        entry_price=pos.entry_price,
                        exit_price=close_price,
                        quantity=pos.quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        hold_days=(current_date - pos.entry_date).days,
                    )
                )
                del open_positions[sym]

        # 2. Gather candidate signals
        candidates: list = []
        for strat_name, strategy in strategies.items():
            weight = strategy_weights.get(strat_name, 0.05)
            for sym in symbols:
                if not allow_multiple_per_symbol and sym in open_positions:
                    continue
                if current_date not in data[sym].index:
                    continue

                df_full = indicators[(strat_name, sym)]
                mask = df_full.index <= current_date
                df_slice = df_full[mask]
                if len(df_slice) < strategy.min_bars():
                    continue

                i = len(df_slice) - 1
                direction, confidence, reason, sl, tgt = strategy.should_enter(df_slice, i)

                if not direction or confidence <= 0.3:
                    continue
                if direction != "BUY":
                    continue

                close_price = float(df_slice["close"].iloc[i])
                atr = (
                    float(df_slice["atr"].iloc[i])
                    if "atr" in df_slice.columns
                    else close_price * 0.02
                )

                if sl <= 0 or tgt <= 0:
                    sl, tgt = _default_sl_tgt(direction, close_price, atr)

                score = confidence * weight
                candidates.append(
                    (score, sym, strat_name, direction,
                     confidence, reason, sl, tgt, close_price, atr)
                )

        # 3. Rank and open positions
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3b. Sector cap filter
        if sector_map:
            deployed_val = sum(
                float(data[p.symbol].loc[current_date, "close"]) * p.quantity
                for p in open_positions.values()
                if current_date in data[p.symbol].index
            )
            sector_exposure = compute_sector_exposure(
                open_positions, sector_map, data, current_date
            )
            candidates = filter_candidates_by_sector(
                candidates, sector_map, sector_exposure,
                cash, deployed_val, position_size_pct,
                max_sector_pct,
            )

        for (
            _score, sym, strat_name, direction, _conf, _reason,
            sl, tgt, entry_price, atr,
        ) in candidates:
            if len(open_positions) >= max_positions:
                break
            if not allow_multiple_per_symbol and sym in open_positions:
                continue

            # Position size
            deployed_val = sum(
                float(data[p.symbol].loc[current_date, "close"]) * p.quantity
                for p in open_positions.values()
                if current_date in data[p.symbol].index
            )
            current_equity = cash + deployed_val
            notional = current_equity * position_size_pct
            quantity = int(notional / entry_price)
            if quantity <= 0:
                log.debug(f"Skipping {sym} — zero or negative quantity: {quantity}")
                continue

            # Apply slippage on entry
            exec_price, entry_txn, entry_slip_cost = fill_entry(
                entry_price, quantity, slippage_pct, direction
            )
            if not check_cash_constraint(cash, quantity, exec_price, entry_txn):
                continue

            total_slippage_acc += entry_slip_cost

            # Open
            cash -= quantity * exec_price + entry_txn
            df_full = indicators[(strat_name, sym)]
            mask = df_full.index <= current_date
            df_slice = df_full[mask]
            entry_idx = len(df_slice) - 1

            open_positions[sym] = SimPosition(
                symbol=sym,
                strategy=strat_name,
                direction=direction,
                quantity=quantity,
                entry_price=exec_price,
                entry_date=current_date,
                entry_idx=entry_idx,
                stop_loss=sl,
                target=tgt,
                entry_atr=atr,
                highest_since_entry=entry_price,
                lowest_since_entry=entry_price,
            )

        # 4. Mark to market
        deployed_val = sum(
            float(data[p.symbol].loc[current_date, "close"]) * p.quantity
            for p in open_positions.values()
            if current_date in data[p.symbol].index
        )
        total_equity = cash + deployed_val
        equity_curve.append((current_date.date(), round(total_equity, 2)))
        daily_deployed.append(deployed_val)

    # ── END_OF_TEST: force close ──
    if open_positions and dates:
        last_date = dates[-1]
        for sym, pos in list(open_positions.items()):
            if last_date not in data[sym].index:
                continue
            df_full = indicators[(pos.strategy, sym)]
            mask = df_full.index <= last_date
            df_slice = df_full[mask]
            if len(df_slice) == 0:
                continue
            i = len(df_slice) - 1
            close_price = float(df_slice["close"].iloc[i])
            exit_slip, exit_txn, exit_slip_cost = fill_exit(
                close_price, pos.quantity, slippage_pct, pos.direction
            )
            cash += pos.quantity * exit_slip - exit_txn
            total_slippage_acc += exit_slip_cost

            pnl, pnl_pct = compute_trade_pnl(
                exit_slip, pos.entry_price, pos.quantity,
                pos.direction, exit_txn
            )

            trade_log.append(
                SimTrade(
                    symbol=pos.symbol,
                    strategy=pos.strategy,
                    direction=pos.direction,
                    entry_date=pos.entry_date.date(),
                    exit_date=last_date.date(),
                    entry_price=pos.entry_price,
                    exit_price=close_price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason="END_OF_TEST",
                    hold_days=(last_date - pos.entry_date).days,
                )
            )
            del open_positions[sym]

        deployed_val = sum(
            float(data[p.symbol].loc[last_date, "close"]) * p.quantity
            for p in open_positions.values()
            if last_date in data[p.symbol].index
        )
        total_equity = cash + deployed_val
        if equity_curve and equity_curve[-1][0] != last_date.date():
            equity_curve.append((last_date.date(), round(total_equity, 2)))
        elif equity_curve:
            equity_curve[-1] = (last_date.date(), round(total_equity, 2))

    # ── compute metrics ─────────────────────
    equities = [e for _, e in equity_curve]
    final_equity = equities[-1] if equities else initial_capital
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25) if len(dates) >= 2 else 0.0

    cagr = compute_cagr(initial_capital, final_equity, years)
    max_dd = max_drawdown(equities) if equities else 0.0
    sharpe, sharpe_valid = compute_equity_sharpe(equities) if equities else (0.0, False)
    ui = ulcer_index(equities) if equities else 0.0
    wm = worst_month(equity_curve) if equity_curve else {}

    tm = compute_trade_metrics(trade_log)
    total_tm = tm["total_trades"]
    total_turnover, turnover_annual = compute_turnover(trade_log, equity_curve, initial_capital, dates)

    cash_utilization_avg = 0.0
    if equities and daily_deployed:
        ratios = [d / e for d, e in zip(daily_deployed, equities) if e > 0]
        cash_utilization_avg = round(np.mean(ratios) * 100, 2) if ratios else 0.0

    total_return_pct = round((final_equity / initial_capital - 1) * 100, 2)

    # Summary print
    print_portfolio_summary(
        cagr, max_dd, sharpe, sharpe_valid, tm["profit_factor"],
        ui, turnover_annual, total_tm, wm,
    )

    return SimResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sharpe_valid=sharpe_valid,
        expectancy=tm["expectancy"],
        total_trades=total_tm,
        win_rate=tm["win_rate"],
        cash_utilization_avg=cash_utilization_avg,
        final_equity=round(final_equity, 2),
        initial_capital=round(initial_capital, 2),
        total_return_pct=total_return_pct,
        profit_factor=tm["profit_factor"],
        ulcer_index=round(ui, 2),
        worst_month_return=wm.get("return_pct", 0.0) if wm else 0.0,
        worst_month_label=wm.get("month", "") if wm else "",
        turnover=total_turnover,
        turnover_annual=turnover_annual,
        total_slippage_cost=round(total_slippage_acc, 2),
    )