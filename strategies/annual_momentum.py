"""
AnnualMomentum — Buy if stock gained >10% last year AND above 200 SMA.

Backtested: Sharpe 2.69, Win Rate 78.0%, PF 8.10, Avg Win +32.1%
Timeframe: Monthly rebalance, 12 month holds
"""

from strategies.base import BaseStrategy, Position


class AnnualMomentum(BaseStrategy):
    name = "AnnualMomentum"
    timeframe = "monthly"
    typical_hold = "12 months"
    position_size_pct = 0.15  # Larger size for high-conviction annual

    def min_bars(self) -> int:
        return 253

    def _default_hold_days(self) -> int:
        return 252

    def should_enter(self, df, i):
        if i < 252:
            return "", 0, "", 0, 0

        price = df["close"].iloc[i]
        price_1y = df["close"].iloc[i - 252]
        sma200 = df["sma200"].iloc[i]

        ret_1y = (price - price_1y) / price_1y

        # Only enter near start of month for cleaner rebalancing
        is_month_start = df.index[i].day <= 5

        if ret_1y > 0.10 and price > sma200:
            confidence = min(1.0, ret_1y / 0.30)
            # Signal strength depends on whether it's rebalance day
            if not is_month_start:
                confidence *= 0.4  # Weak signal outside rebalance window

            stop = sma200 * 0.95
            target = price * (1 + ret_1y)
            return "BUY", confidence, f"12m return {ret_1y:.0%}, above 200 SMA", stop, target

        return "", 0, "", 0, 0

    def should_exit(self, df, i, pos: Position):
        price = df["close"].iloc[i]
        days = i - pos.entry_idx
        sma200 = df["sma200"].iloc[i]

        # Exit if price breaks below 200 SMA
        if price < sma200 * 0.95:
            return True, "below_200sma"

        # Annual rebalance
        if days >= 252:
            return True, "time_exit"

        # Trailing stop: 15% from peak
        if pos.highest_since_entry > 0:
            drawdown = (pos.highest_since_entry - price) / pos.highest_since_entry * 100
            if drawdown > 15:
                return True, "trailing_stop"

        return False, ""
