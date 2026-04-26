"""
DonchianMonthly — Monthly Donchian channel, catches big multi-month trends.

Backtested: Sharpe 1.35, Win Rate 46.8%, PF 4.65, Avg Win +37.6%
Timeframe: Weekly signals, 1-12 month holds

FIX: Now always calculates proper SL and TGT (previously returned 0).
"""

from strategies.base import BaseStrategy, Position


class DonchianMonthly(BaseStrategy):
    name = "DonchianMonthly"
    timeframe = "weekly"
    typical_hold = "1-12 months"

    def min_bars(self) -> int:
        return 200

    def _default_hold_days(self) -> int:
        return 180

    def should_enter(self, df, i):
        weekly = self.compute_weekly(df.iloc[:i+1])
        if len(weekly) < 21:
            return "", 0, "", 0, 0

        price = weekly["close"].iloc[-1]
        high_20w = weekly["high_20w"].iloc[-2] if len(weekly) > 1 else 0
        low_10w = weekly["low_10w"].iloc[-2] if len(weekly) > 1 else 0
        atr_w = weekly["atr_w"].iloc[-1] if "atr_w" in weekly.columns else 0

        if high_20w <= 0 or atr_w <= 0:
            return "", 0, "", 0, 0

        if price > high_20w:
            # BUY breakout — stop at 10-week low, target at 3x ATR
            stop = max(low_10w, price - 3 * atr_w)  # Never zero
            target = price + 3 * atr_w
            return "BUY", 0.85, f"Donchian 20w breakout at {price:.0f}", stop, target

        if price < low_10w:
            # SELL breakdown — stop at 20-week high, target at 3x ATR below
            stop = min(high_20w, price + 3 * atr_w)  # Never zero
            target = price - 3 * atr_w
            return "SELL", 0.85, f"Donchian 10w breakdown at {price:.0f}", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx

        weekly = self.compute_weekly(df.iloc[:i+1])
        if len(weekly) < 11:
            return days >= 252, "time_exit" if days >= 252 else ""

        low_10w = weekly["low_10w"].iloc[-1] if "low_10w" in weekly.columns else 0
        high_20w = weekly["high_20w"].iloc[-1] if "high_20w" in weekly.columns else float('inf')

        if pos.direction == "BUY":
            if low_10w > 0 and price < low_10w:
                return True, "donchian_exit"
        else:
            if price > high_20w:
                return True, "donchian_exit"

        # Max hold 12 months
        if days >= 252:
            return True, "time_exit"

        return False, ""
