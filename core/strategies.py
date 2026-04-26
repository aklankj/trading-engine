"""
Trading strategies — each returns a signal in [-1.0, +1.0].

  +1.0 = strong buy    -1.0 = strong sell
  +0.5 = moderate buy  -0.5 = moderate sell
   0.0 = no signal / flat

Every strategy is a callable class with:
  - name, type, description
  - best_regimes: which market regimes it performs best in
  - min_bars: minimum data points needed
  - __call__(df) -> float signal
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Protocol

from utils.logger import log


class Strategy(Protocol):
    name: str
    strategy_type: str
    description: str
    best_regimes: list[str]
    min_bars: int
    def __call__(self, df: pd.DataFrame) -> float: ...


# ══════════════════════════════════════════════════════════════
# 1. DUAL MOVING AVERAGE CROSSOVER (Trend Following)
# ══════════════════════════════════════════════════════════════

class DualMACrossover:
    name = "Dual MA Crossover"
    strategy_type = "Trend Following"
    description = "20/50 EMA golden/death cross — rides trends, exits on reversal"
    best_regimes = ["Bull", "Bear"]
    min_bars = 55

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"]
        fast = c.ewm(span=20, adjust=False).mean()
        slow = c.ewm(span=50, adjust=False).mean()

        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]

        # Crossover signals
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            return 1.0
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            return -1.0

        # Continuation
        if curr_fast > curr_slow:
            spread = (curr_fast - curr_slow) / curr_slow
            return min(0.7, spread * 20)
        else:
            spread = (curr_slow - curr_fast) / curr_slow
            return max(-0.7, -spread * 20)


# ══════════════════════════════════════════════════════════════
# 2. RSI MEAN REVERSION
# ══════════════════════════════════════════════════════════════

class RSIMeanReversion:
    name = "RSI Mean Reversion"
    strategy_type = "Mean Reversion"
    description = "Buys oversold (RSI<30), sells overbought (RSI>70)"
    best_regimes = ["Sideways", "Recovery"]
    min_bars = 20

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        rsi = self._compute_rsi(df["close"], 14)
        if np.isnan(rsi):
            return 0.0

        if rsi < 25:
            return 1.0
        if rsi < 30:
            return 0.7
        if rsi < 35:
            return 0.3
        if rsi > 75:
            return -1.0
        if rsi > 70:
            return -0.7
        if rsi > 65:
            return -0.3
        return 0.0

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0


# ══════════════════════════════════════════════════════════════
# 3. BOLLINGER BAND SQUEEZE
# ══════════════════════════════════════════════════════════════

class BollingerSqueeze:
    name = "Bollinger Squeeze"
    strategy_type = "Volatility Breakout"
    description = "Detects volatility compression then trades the breakout direction"
    best_regimes = ["Sideways", "HighVol"]
    min_bars = 25

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"]
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()

        upper = sma + 2 * std
        lower = sma - 2 * std
        bandwidth = (upper - lower) / sma

        curr_bw = bandwidth.iloc[-1]
        prev_bw = bandwidth.iloc[-2]
        price = c.iloc[-1]
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]
        curr_mid = sma.iloc[-1]

        # Squeeze breakout
        if curr_bw > prev_bw * 1.3:
            if price > curr_upper:
                return 1.0
            if price < curr_lower:
                return -1.0

        # Mean reversion within bands
        if price > curr_mid + 1.5 * std.iloc[-1]:
            return -0.3
        if price < curr_mid - 1.5 * std.iloc[-1]:
            return 0.3

        return 0.0


# ══════════════════════════════════════════════════════════════
# 4. MACD MOMENTUM
# ══════════════════════════════════════════════════════════════

class MACDMomentum:
    name = "MACD Momentum"
    strategy_type = "Momentum"
    description = "MACD signal line crossover with histogram divergence filter"
    best_regimes = ["Bull", "Recovery"]
    min_bars = 35

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"]
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        curr_macd = macd.iloc[-1]
        prev_macd = macd.iloc[-2]
        curr_sig = signal.iloc[-1]
        prev_sig = signal.iloc[-2]

        # Signal line crossover
        if curr_macd > curr_sig and prev_macd <= prev_sig:
            return 1.0
        if curr_macd < curr_sig and prev_macd >= prev_sig:
            return -1.0

        # Momentum continuation
        hist = curr_macd - curr_sig
        prev_hist = prev_macd - prev_sig
        if hist > 0 and hist > prev_hist:
            return 0.6
        if hist < 0 and hist < prev_hist:
            return -0.6

        return 0.0


# ══════════════════════════════════════════════════════════════
# 5. KAUFMAN ADAPTIVE TREND
# ══════════════════════════════════════════════════════════════

class KaufmanAdaptive:
    name = "Kaufman Adaptive"
    strategy_type = "Adaptive Trend"
    description = "Efficiency ratio auto-adjusts between trending and ranging modes"
    best_regimes = ["Bull", "Bear", "Sideways"]
    min_bars = 30

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"].values
        period = 20

        # Efficiency Ratio
        direction = abs(c[-1] - c[-period - 1])
        volatility = sum(abs(c[-i] - c[-i - 1]) for i in range(1, period + 1))
        er = direction / (volatility + 1e-10)

        # Trend direction
        trend = (c[-1] - c[-period - 1]) / c[-period - 1]

        if er > 0.6:
            return np.clip(np.sign(trend) * 1.0, -1, 1)
        if er > 0.4:
            return np.clip(np.sign(trend) * 0.5, -1, 1)
        if er > 0.25:
            return np.clip(np.sign(trend) * 0.2, -1, 1)

        return 0.0


# ══════════════════════════════════════════════════════════════
# 6. VOLATILITY RISK PARITY
# ══════════════════════════════════════════════════════════════

class VolRiskParity:
    name = "Vol Risk Parity"
    strategy_type = "Risk Management"
    description = "Sizes positions inversely to volatility — reduces exposure in chaos"
    best_regimes = ["HighVol", "Recovery"]
    min_bars = 25

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"]
        returns = c.pct_change().dropna().values

        vol = np.std(returns[-20:]) * np.sqrt(252)
        target_vol = 0.15
        vol_scale = np.clip(target_vol / (vol + 0.01), 0.1, 2.0)

        trend = (c.iloc[-1] - c.iloc[-21]) / c.iloc[-21]
        return float(np.clip(trend * 10 * vol_scale, -1, 1))


# ══════════════════════════════════════════════════════════════
# 7. DONCHIAN BREAKOUT
# ══════════════════════════════════════════════════════════════

class DonchianBreakout:
    name = "Donchian Breakout"
    strategy_type = "Breakout"
    description = "20-day high/low channel breakouts — classic turtle-trader approach"
    best_regimes = ["Bull", "HighVol"]
    min_bars = 25

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        high_20 = df["high"].iloc[-21:-1].max()
        low_20 = df["low"].iloc[-21:-1].min()
        mid = (high_20 + low_20) / 2
        price = df["close"].iloc[-1]

        if price > high_20:
            return 1.0
        if price < low_20:
            return -1.0
        if price > mid:
            return 0.3
        if price < mid:
            return -0.3
        return 0.0


# ══════════════════════════════════════════════════════════════
# 8. MULTI-TIMEFRAME CONSENSUS
# ══════════════════════════════════════════════════════════════

class MultiTFConsensus:
    name = "Multi-TF Consensus"
    strategy_type = "Ensemble"
    description = "Combines 5/10/20/50-day signals — acts only when multiple agree"
    best_regimes = ["Bull", "Bear", "Recovery"]
    min_bars = 55

    def __call__(self, df: pd.DataFrame) -> float:
        if len(df) < self.min_bars:
            return 0.0

        c = df["close"]
        price = c.iloc[-1]

        signals = []
        for period in [5, 10, 20, 50]:
            sma_val = c.rolling(period).mean().iloc[-1]
            signals.append(1.0 if price > sma_val else -1.0)

        consensus = np.mean(signals)

        if consensus > 0.6:
            return 1.0
        if consensus < -0.6:
            return -1.0
        if abs(consensus) > 0.3:
            return float(consensus * 0.5)
        return 0.0


# ══════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════

STRATEGIES: dict[str, Strategy] = {
    "dual_ma": DualMACrossover(),
    "rsi_reversion": RSIMeanReversion(),
    "bollinger_squeeze": BollingerSqueeze(),
    "macd_momentum": MACDMomentum(),
    "kaufman_adaptive": KaufmanAdaptive(),
    "vol_risk_parity": VolRiskParity(),
    "donchian_breakout": DonchianBreakout(),
    "mtf_consensus": MultiTFConsensus(),
}


def run_all_strategies(df: pd.DataFrame) -> dict[str, float]:
    """Run all strategies and return dict of signals."""
    signals = {}
    for key, strat in STRATEGIES.items():
        try:
            signals[key] = strat(df)
        except Exception as e:
            log.warning(f"Strategy {strat.name} failed: {e}")
            signals[key] = 0.0
    return signals
