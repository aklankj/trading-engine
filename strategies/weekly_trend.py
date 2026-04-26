"""
WeeklyTrend — Weekly Donchian breakout, enter on 20-week high, exit on 10-week low.

Backtested: Sharpe 2.28, Win Rate 56.5%, PF 2.51, Avg Win +14.8%
Timeframe: Weekly signals, 4-26 week holds
"""

from strategies.base import BaseStrategy, Position


class WeeklyTrend(BaseStrategy):
    name = "WeeklyTrend"
    timeframe = "weekly"
    typical_hold = "4-26 weeks"

    def min_bars(self) -> int:
        return 200

    def _default_hold_days(self) -> int:
        return 90

    def should_enter(self, df, i):
        # Need weekly data
        weekly = self.compute_weekly(df.iloc[:i+1])
        if len(weekly) < 21:
            return "", 0, "", 0, 0

        price = weekly["close"].iloc[-1]
        high_20w = weekly["high_20w"].iloc[-2] if len(weekly) > 1 else 0
        low_10w = weekly["low_10w"].iloc[-2] if len(weekly) > 1 else 0
        atr_w = weekly["atr_w"].iloc[-1] if "atr_w" in weekly.columns else 0

        if high_20w <= 0 or atr_w <= 0:
            return "", 0, "", 0, 0

        # Breakout above 20-week high
        if price > high_20w:
            stop = low_10w
            target = price + (price - low_10w) * 1.5
            return "BUY", 0.8, f"20-week high breakout at {price:.0f}", stop, target

        # Breakdown below 10-week low
        if price < low_10w:
            stop = high_20w
            target = price - (high_20w - price) * 1.5
            return "SELL", 0.8, f"10-week low breakdown at {price:.0f}", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx

        weekly = self.compute_weekly(df.iloc[:i+1])
        if len(weekly) < 11:
            return days >= 180, "time_exit" if days >= 180 else ""

        low_10w = weekly["low_10w"].iloc[-1] if "low_10w" in weekly.columns else 0
        high_20w = weekly["high_20w"].iloc[-1] if "high_20w" in weekly.columns else float('inf')

        if pos.direction == "BUY":
            if low_10w > 0 and price < low_10w:
                return True, "donchian_exit"
        else:
            if price > high_20w:
                return True, "donchian_exit"

        if days >= 180:
            return True, "time_exit"

        return False, ""
