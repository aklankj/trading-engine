"""
portfolio/simulator.py

Multi-strategy portfolio simulator with shared capital.

Runs all strategies together on a shared capital pool, enforcing
max positions, no negative cash, one position per symbol,
and transaction costs.

Supports execution delay (T+1) via config:
  EXECUTION_DELAY_DAYS: number of trading days to delay entry (0 = same-day)
  EXECUTION_PRICE:      price column to use for delayed entry ("open")

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
    # ── execution pipeline counters ────────
    signals_raw: int = 0
    signals_filtered: int = 0
    signals_executed: int = 0
    signals_skipped_execution: int = 0
    filter_rate: float = 0.0
    execution_rate: float = 0.0


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


def _precompute_next_date_map(
    dates: list[pd.Timestamp],
    delay_days: int,
) -> dict[pd.Timestamp, pd.Timestamp | None]:
    """
    Pre-compute next available trading date for each date, given a delay
    in trading days (not calendar days).

    Parameters
    ----------
    dates : list[pd.Timestamp]
        Sorted list of all trading dates (unified across symbols).
    delay_days : int
        Number of trading days to skip forward.  0 means same-day.

    Returns
    -------
    dict[pd.Timestamp, pd.Timestamp | None]
        For each date, the trading date *delay_days* steps forward,
        or None if insufficient future dates exist.
    """
    result: dict[pd.Timestamp, pd.Timestamp | None] = {}
    for i, d in enumerate(dates):
        future_dates = dates[i + 1:]
        if len(future_dates) >= delay_days:
            result[d] = future_dates[delay_days - 1]
        else:
            result[d] = None
    return result


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

    # ── execution delay setup ───────────────
    exec_delay = _cfg.EXECUTION_DELAY_DAYS
    exec_price_col = _cfg.EXECUTION_PRICE

    # Pre-compute next trading date for each signal date
    next_date_map: dict[pd.Timestamp, pd.Timestamp | None] = {}
    if exec_delay > 0:
        next_date_map = _precompute_next_date_map(dates, exec_delay)

    # Pending entries: map target date -> list of entry dicts
    pending_entries: dict[pd.Timestamp, list[dict]] = {}

    # Execution statistics
    executed_delayed: int = 0
    skipped_no_next_date: int = 0
    skipped_no_data: int = 0
    skipped_no_price: int = 0

    # ── state ───────────────────────────────
    cash = float(initial_capital)
    open_positions: dict[str, SimPosition] = {}
    trade_log: list[SimTrade] = []
    equity_curve: list[tuple[date, float]] = []
    daily_deployed: list[float] = []
    total_slippage_acc = 0.0  # FIX: accumulates everywhere

    # ── execution pipeline counters ────────
    signals_raw: int = 0
    signals_filtered: int = 0
    signals_executed: int = 0
    signals_skipped_execution: int = 0

    # Pending exits: map target date -> list of pending exit requests
    pending_exits: dict[pd.Timestamp, list[dict]] = {}
    pending_exit_symbols: set[str] = set()

    # ── daily loop ──────────────────────────
    for current_date in dates:
        # ══════════════════════════════════════
        # Step 0:  Process pending entries (delayed entry execution)
        # ══════════════════════════════════════
        if exec_delay > 0 and current_date in pending_entries:
            for entry in pending_entries.pop(current_date):
                sym = entry["symbol"]

                # Validate: symbol has data on execution date
                if current_date not in data[sym].index:
                    skipped_no_data += 1
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym} on {current_date}: "
                        f"date not in symbol data"
                    )
                    continue

                # Validate: price column exists
                if exec_price_col not in data[sym].columns:
                    skipped_no_price += 1
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym} on {current_date}: "
                        f"column '{exec_price_col}' missing"
                    )
                    continue

                # Validate: price is valid (not NaN, not zero/negative)
                price_val = data[sym].loc[current_date, exec_price_col]
                if pd.isna(price_val) or price_val <= 0:
                    skipped_no_price += 1
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym} on {current_date}: "
                        f"price={price_val} (NaN/zero)"
                    )
                    continue

                entry_price = float(price_val)
                strat_name = entry["strategy"]
                direction = entry["direction"]
                sl = entry["sl"]
                tgt = entry["tgt"]
                atr = entry["atr"]

                # Re-check position limits (may have changed since signal)
                if len(open_positions) >= max_positions:
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym} on {current_date}: "
                        f"max_positions ({max_positions}) reached"
                    )
                    continue
                if not allow_multiple_per_symbol and sym in open_positions:
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym} on {current_date}: "
                        f"position already open"
                    )
                    continue

                # Position sizing at execution time
                deployed_val = sum(
                    float(data[p.symbol].loc[current_date, "close"]) * p.quantity
                    for p in open_positions.values()
                    if current_date in data[p.symbol].index
                )
                current_equity = cash + deployed_val
                notional = current_equity * position_size_pct
                quantity = int(notional / entry_price)
                if quantity <= 0:
                    signals_skipped_execution += 1
                    log.debug(f"Skipping {sym} — zero or negative quantity: {quantity}")
                    continue

                # Apply slippage on entry
                exec_price, entry_txn, entry_slip_cost = fill_entry(
                    entry_price, quantity, slippage_pct, direction
                )
                if not check_cash_constraint(cash, quantity, exec_price, entry_txn):
                    signals_skipped_execution += 1
                    log.debug(
                        f"Skipped delayed entry for {sym}: cash constraint"
                    )
                    continue

                total_slippage_acc += entry_slip_cost

                # Open position
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
                executed_delayed += 1
                signals_executed += 1

        # ══════════════════════════════════════
        # Step 0.5:  Process pending exits (delayed exit execution)
        # ══════════════════════════════════════
        if exec_delay > 0 and current_date in pending_exits:
            for exit_req in pending_exits.pop(current_date):
                sym = exit_req["symbol"]
                try:
                    pos = open_positions.get(sym)
                    if not pos:
                        # Position already closed — nothing to do
                        continue

                    # Mutation guard: verify position hasn't changed
                    if pos.entry_date != exit_req["entry_date"]:
                        log.debug(
                            f"Skipped pending exit for {sym} on {current_date}: "
                            f"position has changed since queue"
                        )
                        continue

                    # Validate: symbol has data on execution date
                    if current_date not in data[sym].index:
                        skipped_no_data += 1
                        log.debug(
                            f"Skipped pending exit for {sym} on {current_date}: "
                            f"date not in symbol data"
                        )
                        continue

                    # Validate: price column exists
                    if exec_price_col not in data[sym].columns:
                        skipped_no_price += 1
                        log.debug(
                            f"Skipped pending exit for {sym} on {current_date}: "
                            f"column '{exec_price_col}' missing"
                        )
                        continue

                    # Validate: price is valid (not NaN, not zero/negative)
                    price_val = data[sym].loc[current_date, exec_price_col]
                    if pd.isna(price_val) or price_val <= 0:
                        skipped_no_price += 1
                        log.debug(
                            f"Skipped pending exit for {sym} on {current_date}: "
                            f"price={price_val} (NaN/zero)"
                        )
                        continue

                    exit_price = float(price_val)
                    exit_reason = exit_req["reason"]

                    # Execute exit at open price
                    exit_slip, exit_txn, exit_slip_cost = fill_exit(
                        exit_price, pos.quantity, slippage_pct, pos.direction
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
                            exit_price=exit_price,
                            quantity=pos.quantity,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            exit_reason=exit_reason,
                            hold_days=(current_date - pos.entry_date).days,
                        )
                    )
                    del open_positions[sym]
                finally:
                    # Always remove from pending set, regardless of skip/execute path
                    pending_exit_symbols.discard(sym)

        # ══════════════════════════════════════
        # Step 1:  Evaluate exits — queue for next day or execute same-day
        # ══════════════════════════════════════
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
                if exec_delay > 0:
                    # ── Delayed exit: queue for T+1 open ──────────
                    if sym in pending_exit_symbols:
                        # Already queued — skip to avoid double exit
                        continue

                    target_date = next_date_map.get(current_date)
                    if target_date is None:
                        log.debug(
                            f"Skipped exit signal for {sym} on {current_date}: "
                            f"no next trading date available"
                        )
                        continue

                    pending_exits.setdefault(target_date, []).append({
                        "symbol": sym,
                        "reason": exit_reason,
                        "entry_date": pos.entry_date,
                    })
                    pending_exit_symbols.add(sym)
                else:
                    # ── Same-day exit (backward compat: exec_delay=0) ──
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

        # ══════════════════════════════════════
        # Step 2:  Gather candidate signals
        # ══════════════════════════════════════
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

                # Only count as raw signal if it's a real actionable intent
                if not direction or confidence <= 0.3:
                    continue
                if direction != "BUY":
                    continue

                signals_raw += 1

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

        # ══════════════════════════════════════
        # Step 3:  Rank candidates, filter, open / queue
        # ══════════════════════════════════════
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
            # Check position limits early
            if len(open_positions) >= max_positions:
                break
            if not allow_multiple_per_symbol and sym in open_positions:
                continue

            if exec_delay > 0:
                # ── Delayed execution: queue for later ──────────
                target_date = next_date_map.get(current_date)
                if target_date is None:
                    skipped_no_next_date += 1
                    log.debug(
                        f"Skipped signal for {sym} on {current_date}: "
                        f"no next trading date available (delay={exec_delay})"
                    )
                    continue

                # This signal survived all filters — count as filtered
                signals_filtered += 1

                # Position size at entry time (use current equity as estimate)
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

                # Store full signal context for later execution
                pending_entries.setdefault(target_date, []).append({
                    "symbol": sym,
                    "strategy": strat_name,
                    "direction": direction,
                    "confidence": _conf,
                    "reason": _reason,
                    "sl": sl,
                    "tgt": tgt,
                    "atr": atr,
                    "entry_date": current_date,
                })
            else:
                # ── Same-day execution (backward compat) ──────
                # This signal survived all filters — count as filtered
                signals_filtered += 1

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
                signals_executed += 1

        # ══════════════════════════════════════
        # Step 4:  Mark to market
        # ══════════════════════════════════════
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

    # ── compute pipeline rates ─────────────
    filter_rate = round(signals_filtered / signals_raw * 100, 1) if signals_raw > 0 else 0.0
    execution_rate = round(signals_executed / signals_filtered * 100, 1) if signals_filtered > 0 else 0.0

    # Summary print
    print_portfolio_summary(
        cagr, max_dd, sharpe, sharpe_valid, tm["profit_factor"],
        ui, turnover_annual, total_tm, wm,
    )

    # Print execution stats if delay was active
    if exec_delay > 0:
        print(
            f"  Execution: Executed={executed_delayed} | "
            f"Skipped: "
            f"NoData={skipped_no_data} "
            f"NoPrice={skipped_no_price} "
            f"NoNextDate={skipped_no_next_date}"
        )

    # Print signal pipeline stats
    print(
        f"  Signal Pipeline: "
        f"Raw={signals_raw} | "
        f"Filtered={signals_filtered} ({filter_rate}%) | "
        f"Executed={signals_executed} | "
        f"SkippedEx={signals_skipped_execution} | "
        f"Rate={execution_rate}%"
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
        signals_raw=signals_raw,
        signals_filtered=signals_filtered,
        signals_executed=signals_executed,
        signals_skipped_execution=signals_skipped_execution,
        filter_rate=filter_rate,
        execution_rate=execution_rate,
    )
