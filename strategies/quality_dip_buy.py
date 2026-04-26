"""
QualityDipBuy — Buy quality stocks on 15-30% dips from 52-week high.

Backtested: Sharpe 2.73, Win Rate 61.3%, PF 2.30, Avg Win +15.4%
Timeframe: Daily signals, 30-90 day holds
"""

from strategies.base import BaseStrategy, Position


class QualityDipBuy(BaseStrategy):
    name = "QualityDipBuy"
    timeframe = "daily"
    typical_hold = "30-90 days"

    def min_bars(self) -> int:
        return 253

    def _default_hold_days(self) -> int:
        return 90

    def should_enter(self, df, i):
        price = df["close"].iloc[i]
        sma200 = df["sma200"].iloc[i]
        drop = df["drop_from_high"].iloc[i]
        atr = df["atr"].iloc[i]

        if 15 < drop < 35 and price > sma200 and atr > 0:
            confidence = min(1.0, drop / 30)
            stop = price * 0.90
            target = price * 1.20
            return "BUY", confidence, f"Down {drop:.0f}% from 52w high, above 200 SMA", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx
        ret = (price - pos.entry_price) / pos.entry_price * 100

        if ret > 20: return True, "target_hit"
        if ret < -10: return True, "stop_loss"
        if days >= 90: return True, "time_exit"

        # Mean reversion: recovered close to high
        drop = df["drop_from_high"].iloc[i]
        if drop < 5: return True, "recovered"

        return False, ""
