"""
Exit strategy engine — the most impactful improvement.

Manages 4 types of exits for every open position:
  1. Trailing stop (Chandelier Exit — 3x ATR from highest high)
  2. Time-based (cut dead capital after N bars)
  3. Regime-based (exit when regime shifts against your trade)
  4. Partial profit-taking (sell 50% at 1.5x risk, trail the rest)

Usage:
  manager = ExitManager()
  manager.register_trade(trade)
  exit_signal = manager.check_exits(trade_id, current_price, current_atr, regime)
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logger import log
from utils import now_ist


class ExitReason(Enum):
    NONE = "none"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    REGIME_SHIFT = "regime_shift"
    PARTIAL_PROFIT = "partial_profit"
    TARGET_HIT = "target_hit"
    HARD_STOP = "hard_stop"


@dataclass
class ExitSignal:
    should_exit: bool
    reason: ExitReason
    exit_quantity_pct: float    # 0.0-1.0, 1.0 = full exit
    exit_price: float
    message: str


@dataclass
class TradeState:
    """Tracks evolving state of an open position."""
    trade_id: str
    symbol: str
    direction: str               # BUY or SELL
    entry_price: float
    quantity: int
    remaining_quantity: int
    entry_atr: float
    entry_regime: str
    entry_time: datetime
    stop_loss: float
    target: float
    highest_since_entry: float   # For trailing stop (longs)
    lowest_since_entry: float    # For trailing stop (shorts)
    bars_held: int = 0
    partial_taken: bool = False
    trailing_active: bool = False


class ExitManager:
    """
    Manages exits for all open positions.

    Config parameters:
      chandelier_mult:   ATR multiplier for trailing stop (default 3.0)
      max_holding_bars:  Max bars before time exit (default 40 = ~2 months daily)
      partial_rr:        Risk-reward level for partial profit (default 1.5)
      partial_pct:       % to sell at partial profit (default 0.5 = 50%)
    """

    def __init__(
        self,
        chandelier_mult: float = 3.0,
        max_holding_bars: int = 40,
        partial_rr: float = 1.5,
        partial_pct: float = 0.5,
    ):
        self.chandelier_mult = chandelier_mult
        self.max_holding_bars = max_holding_bars
        self.partial_rr = partial_rr
        self.partial_pct = partial_pct
        self.trades: dict[str, TradeState] = {}

    def register_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        entry_atr: float,
        entry_regime: str,
        stop_loss: float,
        target: float,
    ) -> TradeState:
        """Register a new trade for exit management."""
        state = TradeState(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            remaining_quantity=quantity,
            entry_atr=entry_atr,
            entry_regime=entry_regime,
            entry_time=now_ist(),
            stop_loss=stop_loss,
            target=target,
            highest_since_entry=entry_price,
            lowest_since_entry=entry_price,
        )
        self.trades[trade_id] = state
        log.info(f"Exit manager: registered {trade_id} {direction} {symbol} @ {entry_price}")
        return state

    def check_exits(
        self,
        trade_id: str,
        current_price: float,
        current_atr: float,
        current_regime: str,
        current_high: float = None,
        current_low: float = None,
    ) -> ExitSignal:
        """
        Check all exit conditions for a trade.
        Call this every bar (or every 5 min for intraday).

        Priority: Hard stop > Regime shift > Trailing stop > Time exit > Partial profit
        """
        state = self.trades.get(trade_id)
        if not state:
            return ExitSignal(False, ExitReason.NONE, 0, 0, "Trade not found")

        # Update tracking
        state.bars_held += 1
        if current_high:
            state.highest_since_entry = max(state.highest_since_entry, current_high)
        else:
            state.highest_since_entry = max(state.highest_since_entry, current_price)
        if current_low:
            state.lowest_since_entry = min(state.lowest_since_entry, current_low)
        else:
            state.lowest_since_entry = min(state.lowest_since_entry, current_price)

        is_long = state.direction == "BUY"

        # ── 1. Hard stop-loss ─────────────────────────────────
        if is_long and current_price <= state.stop_loss:
            return ExitSignal(
                True, ExitReason.HARD_STOP, 1.0, state.stop_loss,
                f"Hard SL hit at {state.stop_loss:.2f}"
            )
        if not is_long and current_price >= state.stop_loss:
            return ExitSignal(
                True, ExitReason.HARD_STOP, 1.0, state.stop_loss,
                f"Hard SL hit at {state.stop_loss:.2f}"
            )

        # ── 2. Target hit ─────────────────────────────────────
        if is_long and current_price >= state.target:
            return ExitSignal(
                True, ExitReason.TARGET_HIT, 1.0, current_price,
                f"Target hit at {current_price:.2f}"
            )
        if not is_long and current_price <= state.target:
            return ExitSignal(
                True, ExitReason.TARGET_HIT, 1.0, current_price,
                f"Target hit at {current_price:.2f}"
            )

        # ── 3. Regime-based exit ──────────────────────────────
        regime_exit = self._check_regime_exit(state, current_regime)
        if regime_exit.should_exit:
            return regime_exit

        # ── 4. Partial profit-taking ──────────────────────────
        partial_exit = self._check_partial_profit(state, current_price)
        if partial_exit.should_exit:
            return partial_exit

        # ── 5. Trailing stop (Chandelier Exit) ────────────────
        trailing_exit = self._check_trailing_stop(state, current_price, current_atr)
        if trailing_exit.should_exit:
            return trailing_exit

        # ── 6. Time-based exit ────────────────────────────────
        time_exit = self._check_time_exit(state, current_price)
        if time_exit.should_exit:
            return time_exit

        return ExitSignal(False, ExitReason.NONE, 0, 0, "No exit conditions met")

    def _check_trailing_stop(
        self, state: TradeState, price: float, atr: float
    ) -> ExitSignal:
        """Chandelier Exit: trail by N * ATR from the highest high (longs) or lowest low (shorts)."""
        is_long = state.direction == "BUY"
        trail_distance = self.chandelier_mult * atr

        if is_long:
            # Activate trailing only after 1x risk profit
            risk = state.entry_price - state.stop_loss
            if price > state.entry_price + risk:
                state.trailing_active = True

            if state.trailing_active:
                trailing_stop = state.highest_since_entry - trail_distance
                # Never trail below entry stop
                trailing_stop = max(trailing_stop, state.stop_loss)

                if price <= trailing_stop:
                    return ExitSignal(
                        True, ExitReason.TRAILING_STOP, 1.0, price,
                        f"Trailing stop hit: peak {state.highest_since_entry:.2f}, "
                        f"trail at {trailing_stop:.2f}"
                    )
        else:
            risk = state.stop_loss - state.entry_price
            if price < state.entry_price - risk:
                state.trailing_active = True

            if state.trailing_active:
                trailing_stop = state.lowest_since_entry + trail_distance
                trailing_stop = min(trailing_stop, state.stop_loss)

                if price >= trailing_stop:
                    return ExitSignal(
                        True, ExitReason.TRAILING_STOP, 1.0, price,
                        f"Trailing stop hit: trough {state.lowest_since_entry:.2f}, "
                        f"trail at {trailing_stop:.2f}"
                    )

        return ExitSignal(False, ExitReason.NONE, 0, 0, "")

    def _check_regime_exit(self, state: TradeState, current_regime: str) -> ExitSignal:
        """Exit when regime shifts against the trade direction."""
        is_long = state.direction == "BUY"

        # Regime danger mappings
        if is_long and state.entry_regime in ("Bull", "Recovery"):
            if current_regime in ("Bear", "HighVol"):
                return ExitSignal(
                    True, ExitReason.REGIME_SHIFT, 1.0, 0,
                    f"Regime shifted from {state.entry_regime} to {current_regime} — "
                    f"against long position"
                )
        if not is_long and state.entry_regime in ("Bear",):
            if current_regime in ("Bull", "Recovery"):
                return ExitSignal(
                    True, ExitReason.REGIME_SHIFT, 1.0, 0,
                    f"Regime shifted from {state.entry_regime} to {current_regime} — "
                    f"against short position"
                )

        return ExitSignal(False, ExitReason.NONE, 0, 0, "")

    def _check_partial_profit(self, state: TradeState, price: float) -> ExitSignal:
        """Sell partial position when profit reaches 1.5x risk."""
        if state.partial_taken:
            return ExitSignal(False, ExitReason.NONE, 0, 0, "")

        is_long = state.direction == "BUY"
        risk = abs(state.entry_price - state.stop_loss)
        profit_target = risk * self.partial_rr

        if is_long and price >= state.entry_price + profit_target:
            state.partial_taken = True
            # Move stop to breakeven after partial
            state.stop_loss = state.entry_price
            return ExitSignal(
                True, ExitReason.PARTIAL_PROFIT, self.partial_pct, price,
                f"Partial profit at {self.partial_rr}x risk. "
                f"Selling {self.partial_pct:.0%}, stop moved to breakeven."
            )
        if not is_long and price <= state.entry_price - profit_target:
            state.partial_taken = True
            state.stop_loss = state.entry_price
            return ExitSignal(
                True, ExitReason.PARTIAL_PROFIT, self.partial_pct, price,
                f"Partial profit at {self.partial_rr}x risk. "
                f"Selling {self.partial_pct:.0%}, stop moved to breakeven."
            )

        return ExitSignal(False, ExitReason.NONE, 0, 0, "")

    def _check_time_exit(self, state: TradeState, price: float) -> ExitSignal:
        """Exit stale positions that haven't moved meaningfully."""
        if state.bars_held < self.max_holding_bars:
            return ExitSignal(False, ExitReason.NONE, 0, 0, "")

        # Only time-exit if the position isn't significantly profitable
        is_long = state.direction == "BUY"
        pnl_pct = (price / state.entry_price - 1) if is_long else (1 - price / state.entry_price)

        if pnl_pct < 0.05:  # Less than 5% profit after max bars
            return ExitSignal(
                True, ExitReason.TIME_EXIT, 1.0, price,
                f"Time exit after {state.bars_held} bars — "
                f"only {pnl_pct:.1%} profit. Freeing capital."
            )

        return ExitSignal(False, ExitReason.NONE, 0, 0, "")

    def remove_trade(self, trade_id: str):
        """Remove a fully closed trade."""
        self.trades.pop(trade_id, None)

    def get_open_trades(self) -> list[TradeState]:
        """Get all active trades being managed."""
        return list(self.trades.values())
