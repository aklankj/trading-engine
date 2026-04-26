"""
portfolio/simulator.py

Multi-strategy portfolio simulator with shared capital.

Runs all strategies together on a shared capital pool, enforcing
max positions, no negative cash, one position per symbol,
and transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Union

from utils.logger import log
from strategies.base import BaseStrategy, Position
from strategies.registry import CORE_STRATEGIES, STRATEGY_WEIGHTS
from analytics.drawdown import max_drawdown, ulcer_index, worst_month


# ──────────────────────────────────────────
# Data structures
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


def _compute_equity_sharpe(equity_curve: list[float], min_days: int = 30) -> tuple[float, bool]:
    """Sharpe ratio from daily equity-curve returns."""
    if len(equity_curve) < min_days + 1:
        return 0.0, False
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    if len(returns) == 0 or np.std(returns) == 0 or np.isnan(np.std(returns)):
        return 0.0, False
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    return round(sharpe, 2), True


# _compute_max_drawdown replaced by analytics.drawdown.max_drawdown


def _compute_cagr(initial: float, final: float, years: float) -> float:
    """Compound annual growth rate (%)."""
    if years <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return round(((final / initial) ** (1 / years) - 1) * 100, 2)


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
        Slippage applied per trade as fraction of price (e.g. 0.001 for 0.1%).
        Applied on entry (buy: price * (1 + slippage)) and exit (sell: price * (1 - slippage)).
    sector_map : dict[str, str] | None
        Optional mapping of symbol -> sector name for sector exposure cap.
        If provided, MAX_SECTOR_EXPOSURE_PCT from config is enforced.

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

    from config.settings import cfg as _cfg
    sector_map = sector_map or {}
    max_sector_pct = _cfg.MAX_SECTOR_EXPOSURE_PCT

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    # ── pre-compute indicators ──────────────
    indicators: dict[tuple[str, str], pd.DataFrame] = {}
    for strat_name, strategy in strategies.items():
        for sym in symbols:
            df = data[sym]
            # Optional date clipping (we still keep full frame so indices line up)
            indicators[(strat_name, sym)] = strategy.compute_indicators(df.copy())

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

            # Build base.Position for should_exit
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
                # Apply slippage on exit
                if pos.direction == "BUY":
                    # Selling — get slightly less due to slippage
                    exit_slip = close_price * (1 - slippage_pct)
                else:
                    exit_slip = close_price * (1 + slippage_pct)

                exit_txn = BaseStrategy.transaction_cost(exit_slip, pos.quantity, "exit")
                cash += pos.quantity * exit_slip - exit_txn
                slippage_cost = abs(close_price - exit_slip) * pos.quantity

                pnl = (exit_slip - pos.entry_price) * pos.quantity - exit_txn
                pnl_pct = (exit_slip - pos.entry_price) / pos.entry_price * 100
                if pos.direction == "SELL":
                    pnl = (pos.entry_price - exit_slip) * pos.quantity - exit_txn
                    pnl_pct = (pos.entry_price - exit_slip) / pos.entry_price * 100

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
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        exit_reason=exit_reason,
                        hold_days=(current_date - pos.entry_date).days,
                    )
                )
                del open_positions[sym]

        # 2. Gather candidate signals
        candidates = []  # list of (score, sym, strat_name, direction, conf, reason, sl, tgt, price)
        for strat_name, strategy in strategies.items():
            weight = strategy_weights.get(strat_name, 0.05)
            for sym in symbols:
                # Skip if already in a position and we disallow multiples
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
                if direction != "BUY":  # Cash equities — no shorting
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
                    (score, sym, strat_name, direction, confidence, reason, sl, tgt, close_price, atr)
                )

        # 3. Rank and open positions
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3b. Apply sector cap before opening
        if sector_map:
            current_equity_for_sector = cash + sum(
                float(data[p.symbol].loc[current_date, "close"]) * p.quantity
                for p in open_positions.values()
                if current_date in data[p.symbol].index
            )
            # Compute current sector exposure
            sector_exposure: dict[str, float] = {}
            for p in open_positions.values():
                if current_date in data[p.symbol].index:
                    sec = sector_map.get(p.symbol, "Unknown")
                    price_today = float(data[p.symbol].loc[current_date, "close"])
                    sector_exposure[sec] = sector_exposure.get(sec, 0) + p.quantity * price_today

            # Filter candidates: skip if adding would exceed cap
            filtered_candidates = []
            for cand in candidates:
                score, sym, strat_name, direction, confidence, reason, sl, tgt, entry_price, atr = cand
                sec = sector_map.get(sym, "Unknown")
                current_sector_val = sector_exposure.get(sec, 0)
                # Estimate new position value
                deployed = sum(
                    float(data[p.symbol].loc[current_date, "close"]) * p.quantity
                    for p in open_positions.values()
                    if current_date in data[p.symbol].index
                )
                ce = cash + deployed
                new_val = ce * position_size_pct
                new_sector_pct = ((current_sector_val + new_val) / ce * 100) if ce > 0 else 0
                if new_sector_pct <= max_sector_pct * 100:
                    filtered_candidates.append(cand)
                else:
                    log.debug(f"  SKIPPED {sym}: {sec} exposure {new_sector_pct:.0f}% > {max_sector_pct*100:.0f}% cap")
            candidates = filtered_candidates

        for (
            score,
            sym,
            strat_name,
            direction,
            confidence,
            reason,
            sl,
            tgt,
            entry_price,
            atr,
        ) in candidates:
            if len(open_positions) >= max_positions:
                break
            if not allow_multiple_per_symbol and sym in open_positions:
                continue

            # Compute position size
            # Use current equity = cash + value of existing positions at current prices
            deployed = 0.0
            for p in open_positions.values():
                if current_date in data[p.symbol].index:
                    price_today = float(data[p.symbol].loc[current_date, "close"])
                    deployed += p.quantity * price_today

            current_equity = cash + deployed
            notional = current_equity * position_size_pct
            quantity = int(notional / entry_price)
            if quantity <= 0:
                continue

            entry_txn = BaseStrategy.transaction_cost(entry_price, quantity, "entry")
            total_cost = quantity * entry_price + entry_txn
            if total_cost > cash:
                continue  # Insufficient cash — skip

            # Apply slippage on entry
            if direction == "BUY":
                exec_price = entry_price * (1 + slippage_pct)
            else:
                exec_price = entry_price * (1 - slippage_pct)
            entry_txn = BaseStrategy.transaction_cost(exec_price, quantity, "entry")
            total_cost = quantity * exec_price + entry_txn
            if total_cost > cash:
                continue  # Recheck with slippage
            slippage_cost_entry = abs(exec_price - entry_price) * quantity

            # Open
            cash -= total_cost
            df_full = indicators[(strat_name, sym)]
            mask = df_full.index <= current_date
            df_slice = df_full[mask]
            entry_idx = len(df_slice) - 1

            open_positions[sym] = SimPosition(
                symbol=sym,
                strategy=strat_name,
                direction=direction,
                quantity=quantity,
                entry_price=exec_price,  # Use executed price
                entry_date=current_date,
                entry_idx=entry_idx,
                stop_loss=sl,
                target=tgt,
                entry_atr=atr,
                highest_since_entry=entry_price,
                lowest_since_entry=entry_price,
            )

        # 4. Mark to market and record equity
        deployed = 0.0
        for p in open_positions.values():
            if current_date in data[p.symbol].index:
                price_today = float(data[p.symbol].loc[current_date, "close"])
                deployed += p.quantity * price_today

        total_equity = cash + deployed
        equity_curve.append((current_date.date(), round(total_equity, 2)))
        daily_deployed.append(deployed)

    # ── END_OF_TEST: force close all open positions at final date ──
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
            exit_txn = BaseStrategy.transaction_cost(close_price, pos.quantity, "exit")
            cash += pos.quantity * close_price - exit_txn

            pnl = (close_price - pos.entry_price) * pos.quantity - exit_txn
            pnl_pct = (close_price - pos.entry_price) / pos.entry_price * 100
            if pos.direction == "SELL":
                pnl = (pos.entry_price - close_price) * pos.quantity - exit_txn
                pnl_pct = (pos.entry_price - close_price) / pos.entry_price * 100

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
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason="END_OF_TEST",
                    hold_days=(last_date - pos.entry_date).days,
                )
            )
            del open_positions[sym]
        # Re-record final equity after forced closes
        deployed = 0.0
        for p in open_positions.values():
            if last_date in data[p.symbol].index:
                price_today = float(data[p.symbol].loc[last_date, "close"])
                deployed += p.quantity * price_today
        total_equity = cash + deployed
        if equity_curve and equity_curve[-1][0] != last_date.date():
            equity_curve.append((last_date.date(), round(total_equity, 2)))
        elif equity_curve:
            equity_curve[-1] = (last_date.date(), round(total_equity, 2))

    # ── compute turnover ────────────────────
    total_turnover = 0.0
    for t in trade_log:
        entry_notional = t.quantity * t.entry_price
        exit_notional = t.quantity * t.exit_price
        total_turnover += abs(entry_notional) + abs(exit_notional)
    avg_equity = np.mean([e for _, e in equity_curve]) if equity_curve else initial_capital
    years_t = max((dates[-1] - dates[0]).days / 365.25, 1/365.25) if len(dates) >= 2 else 1.0
    turnover_ratio = total_turnover / avg_equity if avg_equity > 0 else 0.0
    turnover_annual = turnover_ratio / years_t if years_t > 0 else 0.0

    # ── compute metrics ─────────────────────
    equities = [e for _, e in equity_curve]
    final_equity = equities[-1] if equities else initial_capital
    years = max((dates[-1] - dates[0]).days / 365.25, 1/365.25) if len(dates) >= 2 else 0.0

    cagr = _compute_cagr(initial_capital, final_equity, years)
    max_dd = max_drawdown(equities) if equities else 0.0
    sharpe, sharpe_valid = _compute_equity_sharpe(equities) if equities else (0.0, False)
    ui = ulcer_index(equities) if equities else 0.0
    wm = worst_month(equity_curve) if equity_curve else {}

    # Trade-level metrics
    if trade_log:
        total_trades = len(trade_log)
        wins = [t for t in trade_log if t.pnl > 0]
        losses = [t for t in trade_log if t.pnl <= 0]
        win_rate = round(len(wins) / total_trades * 100, 1)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        expectancy = round(
            (len(wins) / total_trades) * avg_win
            + (len(losses) / total_trades) * avg_loss,
            2,
        )
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = round(gross_profit / gross_loss, 2)
    else:
        total_trades = 0
        win_rate = 0.0
        expectancy = 0.0
        profit_factor = 0.0

    cash_utilization_avg = 0.0
    if equities and daily_deployed:
        ratios = [d / e for d, e in zip(daily_deployed, equities) if e > 0]
        cash_utilization_avg = round(np.mean(ratios) * 100, 2) if ratios else 0.0

    total_return_pct = round((final_equity / initial_capital - 1) * 100, 2)

    # Compact summary print
    sharpe_str = f"{sharpe:.2f}" if sharpe_valid else "N/A"
    pf_str = "INF" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    turnover_str = f"{turnover_annual:.1f}x/yr" if turnover_annual < 100 else f"{turnover_annual:.0f}x/yr"
    print(
        f"Portfolio: CAGR={cagr:.1f}% | MaxDD={max_dd:.1f}% "
        f"| Sharpe={sharpe_str} | PF={pf_str} | Ulcer={ui:.2f} "
        f"| Turnover={turnover_str}"
    )
    if wm:
        print(f"  Worst Month: {wm['month']} ({wm['return_pct']:.1f}%)")
    if turnover_annual > 3.0 and total_trades > 5:
        print(f"  ⚠️  High turnover ({turnover_annual:.1f}x/yr) increases transaction costs")

    return SimResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sharpe_valid=sharpe_valid,
        expectancy=expectancy,
        total_trades=total_trades,
        win_rate=win_rate,
        cash_utilization_avg=cash_utilization_avg,
        final_equity=round(final_equity, 2),
        initial_capital=round(initial_capital, 2),
        total_return_pct=total_return_pct,
        profit_factor=profit_factor,
        ulcer_index=round(ui, 2),
        worst_month_return=wm.get("return_pct", 0.0) if wm else 0.0,
        worst_month_label=wm.get("month", "") if wm else "",
        turnover=round(total_turnover, 2),
        turnover_annual=round(turnover_annual, 2),
        total_slippage_cost=0.0,
    )
