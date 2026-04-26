"""
WeeklyDaily — Weekly trend direction + daily pullback entry.

Backtested: Sharpe 0.69, Win Rate 46.2%, PF 1.72, Avg Win +8.0%
Timeframe: Daily entry, 15-45 day holds
"""

from strategies.base import BaseStrategy, Position


class WeeklyDaily(BaseStrategy):
    name = "WeeklyDaily"
    timeframe = "daily"
    typical_hold = "15-45 days"

    def min_bars(self) -> int:
        return 200

    def _default_hold_days(self) -> int:
        return 30

    def should_enter(self, df, i):
        price = df["close"].iloc[i]
        sma200 = df["sma200"].iloc[i]
        atr = df["atr"].iloc[i]
        rsi = df["rsi"].iloc[i]

        if atr <= 0:
            return "", 0, "", 0, 0

        # Get weekly trend
        weekly = self.compute_weekly(df.iloc[:i+1])
        if len(weekly) < 27:
            return "", 0, "", 0, 0

        sma13w = weekly["sma13w"].iloc[-1] if "sma13w" in weekly.columns else 0
        sma26w = weekly["sma26w"].iloc[-1] if "sma26w" in weekly.columns else 0

        if sma13w <= 0 or sma26w <= 0:
            return "", 0, "", 0, 0

        weekly_uptrend = sma13w > sma26w

        # Weekly uptrend + daily RSI pullback + above 200 SMA
        if weekly_uptrend and rsi < 40 and price > sma200:
            stop = price - 2.5 * atr
            target = price + 3.5 * atr
            return "BUY", 0.65, f"Weekly uptrend + daily RSI pullback ({rsi:.0f})", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx
        atr = pos.entry_atr if pos.entry_atr > 0 else df["atr"].iloc[i]

        # Trailing stop
        trail = pos.highest_since_entry - 2.5 * atr
        if price < trail:
            return True, "trailing_stop"

        if days >= 45:
            return True, "time_exit"

        return False, ""
