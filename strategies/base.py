"""
Unified Strategy Interface.

Every strategy — hand-coded or LLM-generated — implements this interface.
The SAME code runs in both backtest and live. No divergence possible.

Key methods:
  - should_enter(df, i) → (direction, confidence, reason)
  - should_exit(df, i, position) → (should_exit, reason)
  - signal(df) → SwingSignal  (calls should_enter on last bar)
  - backtest(df) → BacktestResult (loops should_enter/exit over history)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class SwingSignal:
    signal: float          # -1.0 to +1.0
    direction: str         # BUY, SELL, HOLD
    strategy: str
    stop_loss: float = 0
    target: float = 0
    hold_days: int = 0
    confidence: float = 0
    reason: str = ""


@dataclass
class Position:
    direction: str
    entry_price: float
    entry_idx: int
    stop_loss: float
    target: float
    entry_atr: float = 0
    highest_since_entry: float = 0
    lowest_since_entry: float = float('inf')


@dataclass
class BacktestResult:
    strategy: str
    trades: list = field(default_factory=list)       # list of pct returns
    equity_final: float = 100000
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0
    # Gated metrics (only valid above minimum trade counts)
    sharpe: float = 0              # Only valid if total_trades >= 30
    sharpe_valid: bool = False     # Whether Sharpe should be trusted
    profit_factor: float = 0
    max_drawdown: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    avg_hold_days: int = 0
    # New: always-valid metrics
    cagr: float = 0               # Compound annual growth rate
    expectancy: float = 0          # Expected return per trade (%)
    total_return_pct: float = 0    # Total return over period
    years_tested: float = 0        # Duration of test
    equity_curve: list = field(default_factory=list)  # For drawdown analysis
    # Minimum trades for Sharpe validity
    MIN_TRADES_FOR_SHARPE: int = field(default=30, repr=False)


class BaseStrategy(ABC):
    """
    All strategies inherit from this.
    Implement should_enter() and get_exit_params().
    Everything else — signal(), backtest(), indicators — is provided.
    """
    name: str = "BaseStrategy"
    timeframe: str = "daily"
    typical_hold: str = "unknown"
    position_size_pct: float = 0.10  # 10% of equity per trade

    # ── Indicators (computed once, shared by enter/exit) ──

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common indicators to dataframe. Override to add more."""
        df = df.copy()

        # Moving averages
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 52-week high/low
        df["high_52w"] = df["high"].rolling(252).max()
        df["low_52w"] = df["low"].rolling(252).min()

        # Drop from 52w high
        df["drop_from_high"] = (df["high_52w"] - df["close"]) / df["high_52w"] * 100

        # Volume average
        df["vol_avg"] = df["volume"].rolling(20).mean()

        # Volatility
        df["volatility"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        return df

    @staticmethod
    def compute_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """Resample to weekly and add weekly indicators."""
        weekly = df.resample("W").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        if len(weekly) > 20:
            weekly["high_20w"] = weekly["high"].rolling(20).max()
            weekly["low_10w"] = weekly["low"].rolling(10).min()
            weekly["sma13w"] = weekly["close"].rolling(13).mean()
            weekly["sma26w"] = weekly["close"].rolling(26).mean()
            weekly["atr_w"] = (weekly["high"] - weekly["low"]).rolling(10).mean()

        return weekly

    # ── Abstract methods (implement these) ──

    @abstractmethod
    def should_enter(self, df: pd.DataFrame, i: int) -> tuple[str, float, str, float, float]:
        """
        Check if we should enter a position at bar i.

        Args:
            df: DataFrame with indicators computed
            i: current bar index (use .iloc[i])

        Returns:
            (direction, confidence, reason, stop_loss, target)
            direction: "BUY", "SELL", or "" for no entry
            confidence: 0.0 to 1.0
            reason: human-readable string
            stop_loss: price level
            target: price level
        """
        ...

    @abstractmethod
    def should_exit(self, df: pd.DataFrame, i: int, pos: Position) -> tuple[bool, str]:
        """
        Check if we should exit an existing position at bar i.

        Args:
            df: DataFrame with indicators
            i: current bar index
            pos: current Position

        Returns:
            (should_exit, reason)
        """
        ...

    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars of data needed before strategy can generate signals."""
        ...

    # ── Live signal (calls should_enter on last bar) ──

    def signal(self, df: pd.DataFrame) -> SwingSignal:
        """Generate signal for current market state. Used by live engine."""
        if len(df) < self.min_bars():
            return SwingSignal(0, "HOLD", self.name)

        df = self.compute_indicators(df)
        i = len(df) - 1

        direction, confidence, reason, stop_loss, target = self.should_enter(df, i)

        if not direction:
            return SwingSignal(0, "HOLD", self.name)

        signal_val = confidence if direction == "BUY" else -confidence

        return SwingSignal(
            signal=signal_val,
            direction=direction,
            strategy=self.name,
            stop_loss=stop_loss,
            target=target,
            hold_days=self._default_hold_days(),
            confidence=confidence,
            reason=reason,
        )

    def _default_hold_days(self) -> int:
        """Default holding period for the strategy."""
        return 60

    # ── Transaction cost model (Finding #8) ──

    @staticmethod
    def transaction_cost(price: float, quantity: int, direction: str) -> float:
        """
        Indian equity transaction costs (NSE, Zerodha):
        - Brokerage: ₹0 (delivery) or ₹20/order (intraday)
        - STT: 0.1% on sell side (delivery), 0.025% both sides (intraday)
        - Exchange charges: ~0.00345%
        - GST: 18% on brokerage + exchange charges
        - Stamp duty: 0.015% on buy side
        - Sebi charges: 0.0001%

        For swing trades (delivery), total ~0.15% per round trip.
        We use 0.1% per trade (entry + exit) as conservative estimate.
        """
        return price * quantity * 0.001  # 0.1% per trade

    # ── Backtest (loops should_enter/exit over history) ──

    def backtest(self, df: pd.DataFrame, years: int = 10) -> BacktestResult:
        """
        Backtest this strategy on historical data.
        Uses the SAME should_enter/exit logic as live.
        """
        result = BacktestResult(strategy=self.name)

        if len(df) < self.min_bars() + 100:
            return result

        df = self.compute_indicators(df)

        equity = 100000.0
        position = None
        trades = []
        hold_days_list = []
        peak_equity = equity
        curve = [equity]

        start = self.min_bars()

        for i in range(start, len(df)):
            price = df["close"].iloc[i]
            atr = df["atr"].iloc[i] if "atr" in df.columns else price * 0.02

            # Check exit first
            if position is not None:
                # Update tracking
                position.highest_since_entry = max(position.highest_since_entry, price)
                position.lowest_since_entry = min(position.lowest_since_entry, price)

                should_exit, exit_reason = self.should_exit(df, i, position)

                if should_exit:
                    if position.direction == "BUY":
                        ret_pct = (price - position.entry_price) / position.entry_price * 100
                    else:
                        ret_pct = (position.entry_price - price) / position.entry_price * 100

                    trades.append(ret_pct)
                    hold_days_list.append(i - position.entry_idx)
                    pos_size = equity * self.position_size_pct
                    pnl = pos_size * ret_pct / 100
                    # Deduct transaction costs (entry + exit)
                    quantity = max(1, int(pos_size / position.entry_price))
                    cost = self.transaction_cost(position.entry_price, quantity, "entry") + \
                           self.transaction_cost(price, quantity, "exit")
                    equity += pnl - cost
                    peak_equity = max(peak_equity, equity)
                    curve.append(equity)
                    position = None

            # Check entry — reject SELL for cash-equity backtests
            if position is None:
                direction, confidence, reason, sl, tgt = self.should_enter(df, i)

                if direction == "SELL":
                    continue

                if direction and confidence > 0.3 and atr > 0:
                    # Use strategy's SL/TGT, or default ATR-based
                    if sl <= 0:
                        sl = price - 2.5 * atr if direction == "BUY" else price + 2.5 * atr
                    if tgt <= 0:
                        tgt = price + 4 * atr if direction == "BUY" else price - 4 * atr

                    position = Position(
                        direction=direction,
                        entry_price=price,
                        entry_idx=i,
                        stop_loss=sl,
                        target=tgt,
                        entry_atr=atr,
                        highest_since_entry=price,
                        lowest_since_entry=price,
                    )

        # Close any remaining position at last price (with transaction costs)
        if position is not None:
            price = df["close"].iloc[-1]
            if position.direction == "BUY":
                ret_pct = (price - position.entry_price) / position.entry_price * 100
            else:
                ret_pct = (position.entry_price - price) / position.entry_price * 100
            trades.append(ret_pct)
            hold_days_list.append(len(df) - 1 - position.entry_idx)
            pos_size = equity * self.position_size_pct
            pnl = pos_size * ret_pct / 100
            quantity = max(1, int(pos_size / position.entry_price))
            cost = self.transaction_cost(position.entry_price, quantity, "entry") + \
                   self.transaction_cost(price, quantity, "exit")
            equity += pnl - cost
            peak_equity = max(peak_equity, equity)
            curve.append(equity)

        # Calculate metrics
        if not trades:
            return result

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        result.trades = trades
        result.equity_final = round(equity, 2)
        result.total_trades = len(trades)
        result.winners = len(wins)
        result.losers = len(losses)
        result.win_rate = round(len(wins) / len(trades) * 100, 1) if trades else 0

        result.avg_win = round(np.mean(wins), 1) if wins else 0
        result.avg_loss = round(np.mean(losses), 1) if losses else 0
        result.avg_hold_days = round(np.mean(hold_days_list)) if hold_days_list else 0

        # ── Total return ──
        result.total_return_pct = round((equity / 100000 - 1) * 100, 2)

        # ── CAGR (always valid, even with few trades) ──
        trading_days = len(df) - start
        result.years_tested = round(trading_days / 252, 1)
        if result.years_tested > 0 and equity > 0:
            result.cagr = round(
                ((equity / 100000) ** (1 / result.years_tested) - 1) * 100, 2
            )

        # ── Expectancy (always valid — expected % return per trade) ──
        # E = (win_rate * avg_win) + (loss_rate * avg_loss)
        # avg_loss is already negative
        if trades:
            wr = len(wins) / len(trades)
            lr = len(losses) / len(trades)
            result.expectancy = round(
                wr * (np.mean(wins) if wins else 0) + lr * (np.mean(losses) if losses else 0), 2
            )

        # ── Sharpe — GATED at minimum 30 trades ──
        if len(trades) >= result.MIN_TRADES_FOR_SHARPE:
            if np.std(trades) > 0:
                # Annualize using actual trades-per-year, not assumed 12
                trades_per_year = len(trades) / max(result.years_tested, 0.5)
                result.sharpe = round(
                    np.mean(trades) / np.std(trades) * np.sqrt(trades_per_year), 2
                )
                result.sharpe_valid = True
        else:
            # Below threshold — report raw ratio but mark as invalid
            if len(trades) > 1 and np.std(trades) > 0:
                result.sharpe = round(np.mean(trades) / np.std(trades), 2)
            result.sharpe_valid = False

        # ── Profit factor ──
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

        # ── Max drawdown from equity curve (uses the same cost model as main P&L) ──
        if curve:
            curve_peak = curve[0]
            max_dd = 0.0
            for val in curve:
                curve_peak = max(curve_peak, val)
                dd = (curve_peak - val) / curve_peak * 100 if curve_peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown = round(max_dd, 1)
            result.equity_curve = curve

        return result
