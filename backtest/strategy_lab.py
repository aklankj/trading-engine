"""
Strategy Lab — Tests a wide range of strategies across multiple timeframes.

Categories:
1. Momentum (1-12 month lookback)
2. Mean Reversion (weekly/monthly)
3. Trend Following (weekly/monthly)
4. Fundamental + Price combos
5. Multi-timeframe
6. Sector Rotation
7. Dual Momentum (absolute + relative)

Each strategy is self-contained and returns (trades, final_equity).
"""

import sys, time
sys.path.insert(0, "/root/trading-engine")

import numpy as np
import pandas as pd
from backtest.engine import fetch_yfinance_data, TOP10_SYMBOLS, NIFTY50_SYMBOLS
from utils.logger import log


# ═══════════════════════════════════════════════════════════
# MOMENTUM STRATEGIES
# ═══════════════════════════════════════════════════════════

def momentum_12m_1m(df, hold_days=21):
    """
    Classic 12-1 momentum: Buy if 12-month return (excluding last month)
    is positive. Rebalance monthly. The most studied factor in finance.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["ret_12m"] = df["close"].pct_change(252)
    df["ret_1m"] = df["close"].pct_change(21)
    df["mom_12_1"] = df["ret_12m"] - df["ret_1m"]  # 12m minus recent 1m
    df["sma200"] = df["close"].rolling(200).mean()

    # Monthly rebalance points
    monthly = df.resample("M").last().index

    for date in monthly:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 253:
            continue

        row = df.iloc[idx]
        price = row["close"]
        mom = row["mom_12_1"]
        above_200 = price > row["sma200"]

        # Close existing position
        if position:
            ret = (price - position["entry"]) / position["entry"] * 100
            trades.append(ret)
            size = equity * 0.15
            equity += size * ret / 100
            position = None

        # Enter if momentum positive and above 200 SMA
        if mom > 0.05 and above_200:
            position = {"entry": price, "idx": idx}

    return trades, equity


def momentum_6m(df, hold_days=21):
    """
    6-month momentum with monthly rebalance.
    Simpler, catches faster trends.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["ret_6m"] = df["close"].pct_change(126)
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    monthly = df.resample("M").last().index

    for date in monthly:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 253:
            continue

        price = df["close"].iloc[idx]
        mom = df["ret_6m"].iloc[idx]
        above_200 = price > df["sma200"].iloc[idx]
        sma_trend = df["sma50"].iloc[idx] > df["sma200"].iloc[idx]

        if position:
            ret = (price - position["entry"]) / position["entry"] * 100
            trades.append(ret)
            size = equity * 0.15
            equity += size * ret / 100
            position = None

        if mom > 0.03 and above_200 and sma_trend:
            position = {"entry": price, "idx": idx}

    return trades, equity


def dual_momentum(df, hold_days=21):
    """
    Gary Antonacci's Dual Momentum: combine absolute + relative momentum.
    Buy only when both stock momentum AND absolute returns are positive.
    Uses risk-free rate proxy of 6% (Indian FD rate).
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["ret_12m"] = df["close"].pct_change(252)
    df["sma200"] = df["close"].rolling(200).mean()
    risk_free_monthly = 0.06 / 12

    monthly = df.resample("M").last().index

    for date in monthly:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 253:
            continue

        price = df["close"].iloc[idx]
        ret_12m = df["ret_12m"].iloc[idx]
        above_200 = price > df["sma200"].iloc[idx]

        if position:
            ret = (price - position["entry"]) / position["entry"] * 100
            trades.append(ret)
            size = equity * 0.15
            equity += size * ret / 100
            position = None

        # Absolute momentum: 12m return > risk-free rate
        # Relative momentum: above 200 SMA (proxy for relative strength)
        if ret_12m > 0.06 and above_200:
            position = {"entry": price, "idx": idx}

    return trades, equity


# ═══════════════════════════════════════════════════════════
# TREND FOLLOWING (LONGER TIMEFRAME)
# ═══════════════════════════════════════════════════════════

def monthly_trend_follow(df, hold_months=6):
    """
    Monthly trend following: buy when monthly close > 10-month SMA.
    Exit when monthly close < 10-month SMA.
    Meb Faber's classic GTAA approach adapted for Indian stocks.
    """
    trades = []
    equity = 100000
    position = None

    monthly = df.resample("M").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    monthly["sma10"] = monthly["close"].rolling(10).mean()

    for i in range(11, len(monthly)):
        price = monthly["close"].iloc[i]
        sma10 = monthly["sma10"].iloc[i]
        prev_price = monthly["close"].iloc[i-1]
        prev_sma = monthly["sma10"].iloc[i-1]

        if position:
            # Exit when price crosses below 10-month SMA
            if price < sma10 and prev_price >= prev_sma:
                ret = (price - position["entry"]) / position["entry"] * 100
                trades.append(ret)
                size = equity * 0.15
                equity += size * ret / 100
                position = None

        if not position:
            # Enter when price crosses above 10-month SMA
            if price > sma10 and prev_price <= prev_sma:
                position = {"entry": price, "idx": i}

    # Close any remaining position
    if position:
        price = monthly["close"].iloc[-1]
        ret = (price - position["entry"]) / position["entry"] * 100
        trades.append(ret)
        equity += equity * 0.15 * ret / 100

    return trades, equity


def donchian_monthly(df, entry_weeks=20, exit_weeks=10):
    """
    Monthly Donchian channel breakout (Turtle Trading adapted).
    Enter on 20-week high breakout, exit on 10-week low.
    Hold as long as trend persists — can be months to years.
    """
    trades = []
    equity = 100000
    position = None

    weekly = df.resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    weekly["high_20w"] = weekly["high"].rolling(entry_weeks).max()
    weekly["low_10w"] = weekly["low"].rolling(exit_weeks).min()
    weekly["atr"] = (weekly["high"] - weekly["low"]).rolling(10).mean()

    for i in range(entry_weeks + 1, len(weekly)):
        price = weekly["close"].iloc[i]
        high20 = weekly["high_20w"].iloc[i-1]
        low10 = weekly["low_10w"].iloc[i-1]

        if position:
            if price < low10:
                ret = (price - position["entry"]) / position["entry"] * 100
                trades.append(ret)
                size = equity * 0.10
                equity += size * ret / 100
                position = None

        if not position:
            if price > high20:
                position = {"entry": price, "idx": i}

    if position:
        price = weekly["close"].iloc[-1]
        ret = (price - position["entry"]) / position["entry"] * 100
        trades.append(ret)
        equity += equity * 0.10 * ret / 100

    return trades, equity


# ═══════════════════════════════════════════════════════════
# MEAN REVERSION (LONGER TIMEFRAME)
# ═══════════════════════════════════════════════════════════

def mean_reversion_monthly(df, hold_days=90):
    """
    Monthly mean reversion: buy when stock drops >20% below
    its 12-month moving average but is still in long-term uptrend
    (above 3-year MA). Contrarian value approach.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["sma252"] = df["close"].rolling(252).mean()    # 1-year MA
    df["sma756"] = df["close"].rolling(756).mean()    # 3-year MA
    df["dist_from_1y"] = (df["close"] - df["sma252"]) / df["sma252"] * 100

    for i in range(757, len(df)):
        price = df["close"].iloc[i]
        dist = df["dist_from_1y"].iloc[i]
        above_3y = price > df["sma756"].iloc[i]

        if position:
            days = i - position["idx"]
            ret = (price - position["entry"]) / position["entry"] * 100

            exit = False
            if ret > 25: exit = True       # 25% profit target
            elif ret < -15: exit = True    # 15% stop loss
            elif days >= 180: exit = True   # 6 month max hold
            elif dist > -5: exit = True    # Reverted close to mean

            if exit:
                trades.append(ret)
                size = equity * 0.10
                equity += size * ret / 100
                position = None

        if not position:
            # Buy when >20% below 1-year MA but above 3-year MA
            if dist < -20 and above_3y:
                position = {"entry": price, "idx": i}

    return trades, equity


def bollinger_monthly(df, hold_days=60):
    """
    Monthly Bollinger band mean reversion.
    Buy at lower band on monthly chart, sell at upper band.
    """
    trades = []
    equity = 100000
    position = None

    monthly = df.resample("M").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    monthly["sma20"] = monthly["close"].rolling(20).mean()
    monthly["std20"] = monthly["close"].rolling(20).std()
    monthly["upper"] = monthly["sma20"] + 2 * monthly["std20"]
    monthly["lower"] = monthly["sma20"] - 2 * monthly["std20"]

    for i in range(21, len(monthly)):
        price = monthly["close"].iloc[i]
        lower = monthly["lower"].iloc[i]
        upper = monthly["upper"].iloc[i]
        sma = monthly["sma20"].iloc[i]

        if position:
            if price >= upper or price >= sma * 1.1:
                ret = (price - position["entry"]) / position["entry"] * 100
                trades.append(ret)
                size = equity * 0.10
                equity += size * ret / 100
                position = None

        if not position:
            if price <= lower:
                position = {"entry": price, "idx": i}

    if position:
        price = monthly["close"].iloc[-1]
        ret = (price - position["entry"]) / position["entry"] * 100
        trades.append(ret)
        equity += equity * 0.10 * ret / 100

    return trades, equity


# ═══════════════════════════════════════════════════════════
# MULTI-TIMEFRAME STRATEGIES
# ═══════════════════════════════════════════════════════════

def weekly_daily_combo(df, hold_days=30):
    """
    Weekly trend + daily entry: Identify trend on weekly chart,
    enter on daily pullback. Best of both worlds.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    # Weekly trend
    weekly = df.resample("W").agg({"close": "last"}).dropna()
    weekly["sma13w"] = weekly["close"].rolling(13).mean()
    weekly["sma26w"] = weekly["close"].rolling(26).mean()
    weekly["trend"] = np.where(weekly["sma13w"] > weekly["sma26w"], 1, -1)

    # RSI for daily entry
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    for i in range(201, len(df)):
        date = df.index[i]
        price = df["close"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        above_50 = price > df["sma50"].iloc[i]
        above_200 = price > df["sma200"].iloc[i]

        # Get weekly trend
        week_date = date - pd.Timedelta(days=date.weekday())
        closest_week = weekly.index[weekly.index <= date]
        if len(closest_week) == 0:
            continue
        weekly_trend = weekly.loc[closest_week[-1], "trend"] if closest_week[-1] in weekly.index else 0

        if position:
            days = i - position["idx"]
            position["highest"] = max(position["highest"], price)
            trail = position["highest"] - 2.5 * position["entry_atr"]

            exit = False
            if price < trail: exit = True
            elif days >= 45: exit = True

            if exit:
                ret = (price - position["entry"]) / position["entry"] * 100
                trades.append(ret)
                size = equity * 0.10
                equity += size * ret / 100
                position = None

        if not position and atr > 0:
            # Weekly uptrend + daily pullback (RSI < 40) + above 200 SMA
            if weekly_trend == 1 and rsi < 40 and above_200:
                position = {"entry": price, "idx": i, "highest": price, "entry_atr": atr}

    return trades, equity


def all_weather_adaptive(df, hold_days=60):
    """
    Adapts strategy to market regime:
    - Bull (above 200 SMA, rising): momentum — buy breakouts
    - Bear (below 200 SMA, falling): mean reversion — buy extreme dips
    - Sideways: do nothing
    
    This is the regime-adaptive approach done properly.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["high_52w"] = df["high"].rolling(252).max()
    df["low_52w"] = df["low"].rolling(252).min()
    df["ret_20d"] = df["close"].pct_change(20)
    df["vol_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    for i in range(253, len(df)):
        price = df["close"].iloc[i]
        sma50 = df["sma50"].iloc[i]
        sma200 = df["sma200"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        high52 = df["high_52w"].iloc[i]
        drop_pct = (high52 - price) / high52 * 100
        vol = df["vol_20d"].iloc[i]

        # Determine regime
        if price > sma200 and sma50 > sma200:
            regime = "bull"
        elif price < sma200 and sma50 < sma200:
            regime = "bear"
        else:
            regime = "sideways"

        if position:
            days = i - position["idx"]
            position["highest"] = max(position["highest"], price)

            exit = False
            if regime == "bull" and position["regime"] == "bull":
                trail = position["highest"] - 2.5 * position["entry_atr"]
                if price < trail: exit = True
                elif days >= 90: exit = True
            elif regime == "bear" and position["regime"] == "bear":
                ret_so_far = (price - position["entry"]) / position["entry"] * 100
                if ret_so_far > 15: exit = True
                elif ret_so_far < -10: exit = True
                elif days >= 60: exit = True
            else:
                if days >= 30: exit = True

            if exit:
                ret = (price - position["entry"]) / position["entry"] * 100
                trades.append(ret)
                size = equity * 0.10
                equity += size * ret / 100
                position = None

        if not position and atr > 0:
            if regime == "bull":
                # Momentum: buy on pullback to 50 SMA in uptrend
                if price < sma50 * 1.02 and price > sma200 and rsi < 45:
                    position = {"entry": price, "idx": i, "highest": price,
                               "entry_atr": atr, "regime": "bull"}

            elif regime == "bear":
                # Mean reversion: buy extreme oversold
                if drop_pct > 25 and rsi < 25 and vol < 0.5:
                    position = {"entry": price, "idx": i, "highest": price,
                               "entry_atr": atr, "regime": "bear"}

    return trades, equity


# ═══════════════════════════════════════════════════════════
# LONG-TERM STRATEGIES (1+ YEAR HORIZON)
# ═══════════════════════════════════════════════════════════

def annual_momentum_rebalance(df, hold_months=12):
    """
    Annual rebalance: buy if stock gained >10% last year AND is above
    200 SMA. Hold for 1 year. Simplest possible momentum.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["ret_1y"] = df["close"].pct_change(252)
    df["sma200"] = df["close"].rolling(200).mean()

    # Annual rebalance at start of each year
    yearly = df.resample("YS").first().index

    for date in yearly:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 253:
            continue

        price = df["close"].iloc[idx]
        ret_1y = df["ret_1y"].iloc[idx]
        above_200 = price > df["sma200"].iloc[idx]

        if position:
            ret = (price - position["entry"]) / position["entry"] * 100
            trades.append(ret)
            size = equity * 0.20
            equity += size * ret / 100
            position = None

        if ret_1y > 0.10 and above_200:
            position = {"entry": price}

    return trades, equity


def buy_and_hold_200sma(df, hold_days=999):
    """
    Simplest trend filter: be invested when above 200 SMA,
    go to cash when below. Monthly check.
    Benchmark to see if any strategy beats this.
    """
    trades = []
    equity = 100000
    position = None

    df = df.copy()
    df["sma200"] = df["close"].rolling(200).mean()

    monthly = df.resample("M").last().index

    for date in monthly:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 201:
            continue

        price = df["close"].iloc[idx]
        above = price > df["sma200"].iloc[idx]

        if position and not above:
            ret = (price - position["entry"]) / position["entry"] * 100
            trades.append(ret)
            size = equity * 0.20
            equity += size * ret / 100
            position = None

        if not position and above:
            position = {"entry": price}

    if position:
        price = df["close"].iloc[-1]
        ret = (price - position["entry"]) / position["entry"] * 100
        trades.append(ret)
        equity += equity * 0.20 * ret / 100

    return trades, equity


# ═══════════════════════════════════════════════════════════
# RUN ALL STRATEGIES
# ═══════════════════════════════════════════════════════════

ALL_STRATEGIES = {
    # Previous winners (for comparison)
    "WeeklyTrend": None,        # Already tested
    "QualityDipBuy": None,      # Already tested

    # Momentum
    "Mom_12m_1m": momentum_12m_1m,
    "Mom_6m": momentum_6m,
    "DualMomentum": dual_momentum,

    # Trend Following
    "MonthlyTrend": monthly_trend_follow,
    "DonchianMonthly": donchian_monthly,

    # Mean Reversion
    "MeanRev_Monthly": mean_reversion_monthly,
    "Bollinger_Monthly": bollinger_monthly,

    # Multi-Timeframe
    "Weekly+Daily": weekly_daily_combo,
    "AllWeather": all_weather_adaptive,

    # Long-term
    "AnnualMomentum": annual_momentum_rebalance,
    "BuyHold_200SMA": buy_and_hold_200sma,
}


def run_strategy_lab(symbols=None, years=10):
    """Run all strategies and produce comparison."""
    if symbols is None:
        symbols = TOP10_SYMBOLS

    strategies = {k: v for k, v in ALL_STRATEGIES.items() if v is not None}

    print("=" * 95)
    print(f"  STRATEGY LAB — {len(symbols)} stocks, {years} years, {len(strategies)} strategies")
    print("=" * 95)
    print(f"{'Strategy':20s} {'Trades':>6s} {'WinRate':>8s} {'AvgRet':>8s} {'Sharpe':>8s} {'AvgWin':>8s} {'AvgLoss':>8s} {'PF':>6s}  Verdict")
    print("-" * 95)

    results = {}

    for strat_name, strat_func in strategies.items():
        all_trades = []
        all_equity = []

        for sym in symbols:
            df = fetch_yfinance_data(sym, years=years)
            if df.empty or len(df) < 500:
                continue

            try:
                trades_list, final_eq = strat_func(df)
                all_trades.extend(trades_list)
                all_equity.append(final_eq)
            except Exception as e:
                pass
            time.sleep(0.3)

        if not all_trades:
            print(f"  {strat_name:18s}  — no trades —")
            continue

        wins = [t for t in all_trades if t > 0]
        losses = [t for t in all_trades if t <= 0]
        win_rate = len(wins) / len(all_trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_return = np.mean([(e / 100000 - 1) * 100 for e in all_equity])

        if len(all_trades) > 1 and np.std(all_trades) > 0:
            # Annualize based on avg holding period
            sharpe = np.mean(all_trades) / np.std(all_trades) * np.sqrt(12)
        else:
            sharpe = 0

        verdict = ""
        if sharpe > 1.5 and pf > 1.8: verdict = "🏆 EXCELLENT"
        elif sharpe > 1.0 and pf > 1.5: verdict = "✅ STRONG"
        elif sharpe > 0.5 and pf > 1.2: verdict = "✅ GOOD"
        elif sharpe > 0 and pf > 1.0: verdict = "🟡 OK"
        else: verdict = "❌ WEAK"

        results[strat_name] = {
            "trades": len(all_trades),
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_return, 1),
            "sharpe": round(sharpe, 2),
            "avg_win": round(avg_win, 1),
            "avg_loss": round(avg_loss, 1),
            "profit_factor": round(pf, 2),
            "verdict": verdict,
        }

        print(
            f"  {strat_name:18s} {len(all_trades):6d} {win_rate:7.1f}% "
            f"{avg_return:7.1f}% {sharpe:7.2f} {avg_win:7.1f}% "
            f"{avg_loss:7.1f}% {pf:5.2f}  {verdict}"
        )

    print("=" * 95)

    # Rank and recommend
    ranked = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n📊 RANKING (by Sharpe ratio):")
    for i, (name, data) in enumerate(ranked):
        print(f"  {i+1}. {name:20s} Sharpe={data['sharpe']:.2f} PF={data['profit_factor']:.2f} WR={data['win_rate']:.0f}% {data['verdict']}")

    print("\n💡 RECOMMENDATION:")
    top3 = [name for name, data in ranked[:3] if data["sharpe"] > 0.5]
    if top3:
        print(f"  Use as core strategies: {', '.join(top3)}")
        print(f"  Combine with WeeklyTrend (Sharpe=2.28) and QualityDipBuy (Sharpe=2.73)")
    else:
        print("  No new strategies beat the threshold. Stick with WeeklyTrend + QualityDipBuy.")

    return results


if __name__ == "__main__":
    run_strategy_lab(years=10)
