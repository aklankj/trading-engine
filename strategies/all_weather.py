"""
AllWeather — Regime-adaptive: momentum in bull, mean reversion in bear.

Backtested: Sharpe 0.84, Win Rate 52.7%, PF 1.95, Avg Win +9.3%
Timeframe: Daily signals, 30-90 day holds
"""

import numpy as np
from strategies.base import BaseStrategy, Position


class AllWeather(BaseStrategy):
    name = "AllWeather"
    timeframe = "daily"
    typical_hold = "30-90 days"

    def min_bars(self) -> int:
        return 253

    def _default_hold_days(self) -> int:
        return 60

    def should_enter(self, df, i):
        price = df["close"].iloc[i]
        sma50 = df["sma50"].iloc[i]
        sma200 = df["sma200"].iloc[i]
        atr = df["atr"].iloc[i]
        rsi = df["rsi"].iloc[i]
        drop = df["drop_from_high"].iloc[i]
        vol = df["volatility"].iloc[i]

        if atr <= 0:
            return "", 0, "", 0, 0

        # Bull regime: momentum pullback
        if price > sma200 and sma50 > sma200:
            if price < sma50 * 1.02 and rsi < 45:
                stop = price - 2.5 * atr
                target = price + 4 * atr
                return "BUY", 0.7, f"Bull pullback: RSI={rsi:.0f}, near 50 SMA", stop, target

        # Bear regime: extreme oversold mean reversion
        elif price < sma200 and sma50 < sma200:
            if drop > 25 and rsi < 25 and vol < 0.5:
                stop = price * 0.90
                target = price * 1.15
                return "BUY", 0.6, f"Bear extreme oversold: RSI={rsi:.0f}, down {drop:.0f}%", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx
        sma50 = df["sma50"].iloc[i]
        sma200 = df["sma200"].iloc[i]
        atr = df["atr"].iloc[i] if df["atr"].iloc[i] > 0 else pos.entry_atr

        # Determine current regime
        is_bull = price > sma200 and sma50 > sma200

        if is_bull:
            # Trailing stop in bull
            trail = pos.highest_since_entry - 2.5 * pos.entry_atr
            if price < trail:
                return True, "trailing_stop"
            if days >= 90:
                return True, "time_exit"
        else:
            # Tighter exits in bear
            ret = (price - pos.entry_price) / pos.entry_price * 100
            if ret > 15:
                return True, "target_hit"
            if ret < -10:
                return True, "stop_loss"
            if days >= 60:
                return True, "time_exit"

        return False, ""
