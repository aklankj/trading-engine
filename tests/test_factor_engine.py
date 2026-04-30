"""
tests/test_factor_engine.py

Unit tests for the Factor Engine.
Uses synthetic data — no network access needed.

Run: python -m pytest tests/test_factor_engine.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ─── Fixtures ─────────────────────────────────────────────────

def make_price_df(
    returns: list[float],
    start_price: float = 100.0,
    start_date: str = "2014-01-01",
) -> pd.DataFrame:
    """Generate a daily price DataFrame from a list of daily returns."""
    dates = pd.bdate_range(start=start_date, periods=len(returns))
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * len(prices),
    }, index=dates)
    return df


def make_universe(n_stocks: int = 50, n_days: int = 300) -> dict[str, pd.DataFrame]:
    """Create a synthetic universe with varying momentum."""
    np.random.seed(42)
    universe = {}
    for i in range(n_stocks):
        # Each stock has a different drift (momentum signal)
        drift = 0.0002 * (i - n_stocks // 2)  # Range: -0.005 to +0.005 daily
        noise = np.random.normal(0, 0.02, n_days)
        returns = drift + noise
        returns[0] = 0
        universe[f"STOCK{i:03d}"] = make_price_df(returns, start_price=100 + i)
    return universe


def make_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Create synthetic fundamental data."""
    np.random.seed(123)
    fund = {}
    for i, sym in enumerate(symbols):
        fund[sym] = {
            "symbol": sym,
            "ROE": np.random.uniform(5, 30),
            "ROCE": np.random.uniform(5, 35),
            "Debt to equity": np.random.uniform(0, 2),
            "profit_growth_5 Years": np.random.uniform(-5, 25),
            "OPM": np.random.uniform(5, 30),
            "Stock P/E": np.random.uniform(5, 60),
            "Book Value": np.random.uniform(50, 500),
            "Current Price": 100 + i,
            "Market Cap": np.random.uniform(1000, 100000),
            "Dividend Yield": np.random.uniform(0, 4),
        }
    return fund


# ─── Tests: Momentum Factor ──────────────────────────────────

class TestMomentumFactor:
    def test_basic_momentum(self):
        from factors.momentum import MomentumFactor
        mf = MomentumFactor()

        # Stock A: strong uptrend, Stock B: strong downtrend
        universe = {
            "UP": make_price_df([0] + [0.003] * 299),    # ~2.4x in 300 days
            "DOWN": make_price_df([0] + [-0.002] * 299),  # ~0.55x in 300 days
            "FLAT": make_price_df([0] * 300),
        }

        raw = mf.compute_raw(universe)
        assert "UP" in raw
        assert "DOWN" in raw
        assert raw["UP"] > raw["DOWN"]
        assert raw["UP"] > 0
        assert raw["DOWN"] < 0

    def test_momentum_scoring(self):
        from factors.momentum import MomentumFactor
        mf = MomentumFactor()
        universe = make_universe(50)

        scores = mf.score(universe)
        assert len(scores) >= 10

        # Highest z-score should be the best momentum stock
        best = max(scores.values(), key=lambda s: s.z_score)
        worst = min(scores.values(), key=lambda s: s.z_score)
        assert best.z_score > worst.z_score
        assert best.percentile > 80
        assert worst.percentile < 20

    def test_skip_insufficient_data(self):
        from factors.momentum import MomentumFactor
        mf = MomentumFactor()

        # Only 50 days of data — should be skipped
        short = {"SHORT": make_price_df([0] * 50)}
        raw = mf.compute_raw(short)
        assert len(raw) == 0


# ─── Tests: Quality Factor ───────────────────────────────────

class TestQualityFactor:
    def test_quality_scoring(self):
        from factors.quality import QualityFactor
        qf = QualityFactor()

        symbols = [f"STOCK{i:03d}" for i in range(30)]
        fund = make_fundamentals(symbols)

        # Make one stock clearly high quality
        fund["STOCK000"]["ROE"] = 30
        fund["STOCK000"]["ROCE"] = 35
        fund["STOCK000"]["Debt to equity"] = 0.1
        fund["STOCK000"]["profit_growth_5 Years"] = 25
        fund["STOCK000"]["OPM"] = 28

        # Make one stock clearly low quality
        fund["STOCK029"]["ROE"] = 2
        fund["STOCK029"]["ROCE"] = 3
        fund["STOCK029"]["Debt to equity"] = 3.0
        fund["STOCK029"]["profit_growth_5 Years"] = -10
        fund["STOCK029"]["OPM"] = 3

        raw = qf.compute_raw({}, fund)
        assert raw["STOCK000"] > raw["STOCK029"]

    def test_quality_needs_fundamentals(self):
        from factors.quality import QualityFactor
        qf = QualityFactor()
        raw = qf.compute_raw({}, None)
        assert len(raw) == 0


# ─── Tests: Value Factor ─────────────────────────────────────

class TestValueFactor:
    def test_value_scoring(self):
        from factors.value import ValueFactor
        vf = ValueFactor()

        symbols = [f"STOCK{i:03d}" for i in range(20)]
        fund = make_fundamentals(symbols)

        # Make one stock very cheap
        fund["STOCK000"]["Stock P/E"] = 5     # High earnings yield
        fund["STOCK000"]["Book Value"] = 400
        fund["STOCK000"]["Current Price"] = 100

        # Make one stock very expensive
        fund["STOCK019"]["Stock P/E"] = 100
        fund["STOCK019"]["Book Value"] = 50
        fund["STOCK019"]["Current Price"] = 500

        raw = vf.compute_raw({}, fund)
        assert raw["STOCK000"] > raw["STOCK019"]


# ─── Tests: Composite Engine ─────────────────────────────────

class TestCompositeEngine:
    def test_ranking(self):
        from factors.composite import CompositeFactorEngine, momentum_quality_preset

        universe = make_universe(50)
        symbols = list(universe.keys())
        fund = make_fundamentals(symbols)

        engine = CompositeFactorEngine(
            factors=momentum_quality_preset(),
            top_n=10,
        )

        ranked = engine.rank_universe(universe, fund)
        assert len(ranked) >= 10

        # Ranks should be sequential
        for i, cs in enumerate(ranked):
            assert cs.rank == i + 1

        # First should have highest composite z
        assert ranked[0].composite_z >= ranked[-1].composite_z

    def test_portfolio_selection(self):
        from factors.composite import CompositeFactorEngine, momentum_quality_preset

        universe = make_universe(50)
        symbols = list(universe.keys())
        fund = make_fundamentals(symbols)

        engine = CompositeFactorEngine(
            factors=momentum_quality_preset(),
            top_n=10,
        )

        selected = engine.select_portfolio(universe, fund)
        assert len(selected) == 10
        assert all(cs.selected for cs in selected)

    def test_equal_weights(self):
        from factors.composite import CompositeFactorEngine, momentum_quality_preset

        universe = make_universe(50)
        symbols = list(universe.keys())
        fund = make_fundamentals(symbols)

        engine = CompositeFactorEngine(
            factors=momentum_quality_preset(),
            top_n=10,
            weighting="equal",
        )

        selected = engine.select_portfolio(universe, fund)
        weights = engine.compute_weights(selected)

        assert len(weights) == 10
        assert abs(sum(weights.values()) - 1.0) < 0.01
        for w in weights.values():
            assert abs(w - 0.1) < 0.001

    def test_pure_momentum_without_fundamentals(self):
        """Momentum-only should work without fundamental data."""
        from factors.composite import CompositeFactorEngine, pure_momentum_preset

        universe = make_universe(50)
        engine = CompositeFactorEngine(
            factors=pure_momentum_preset(),
            top_n=10,
        )

        ranked = engine.rank_universe(universe, fundamental_data=None)
        assert len(ranked) >= 10


# ─── Tests: Backtest ──────────────────────────────────────────

class TestFactorBacktest:
    def test_basic_backtest(self):
        from factors.composite import CompositeFactorEngine, pure_momentum_preset
        from factors.backtest import run_factor_backtest

        # Use 3 years of data for a quick test
        universe = make_universe(30, n_days=756)

        engine = CompositeFactorEngine(
            factors=pure_momentum_preset(),
            top_n=5,
            min_avg_volume=0,  # Synthetic data
            min_market_cap=0,
        )

        result = run_factor_backtest(
            price_data=universe,
            engine=engine,
            initial_capital=100_000,
        )

        assert result.start_date != ""
        assert result.end_date != ""
        assert result.total_rebalances > 0
        assert len(result.equity_curve) > 0
        assert result.equity_curve[-1]["equity"] > 0

    def test_backtest_produces_returns(self):
        """Verify the backtest produces non-zero returns."""
        from factors.composite import CompositeFactorEngine, pure_momentum_preset
        from factors.backtest import run_factor_backtest

        universe = make_universe(30, n_days=756)

        engine = CompositeFactorEngine(
            factors=pure_momentum_preset(),
            top_n=5,
            min_avg_volume=0,
            min_market_cap=0,
        )

        result = run_factor_backtest(
            price_data=universe,
            engine=engine,
            initial_capital=100_000,
        )

        # Should have some non-zero return
        assert result.total_return_pct != 0
        # Should have computed CAGR
        assert result.cagr_pct is not None


# ─── Tests: Base Factor Z-Scoring ─────────────────────────────

class TestZScoring:
    def test_z_scores_centered(self):
        from factors.momentum import MomentumFactor
        mf = MomentumFactor()
        universe = make_universe(50)

        scores = mf.score(universe)
        z_values = [s.z_score for s in scores.values()]

        # Mean of z-scores should be near 0
        assert abs(np.mean(z_values)) < 0.3  # Not exactly 0 due to winsorizing

    def test_percentiles_span_range(self):
        from factors.momentum import MomentumFactor
        mf = MomentumFactor()
        universe = make_universe(50)

        scores = mf.score(universe)
        percentiles = [s.percentile for s in scores.values()]

        assert min(percentiles) < 5
        assert max(percentiles) > 95

    def test_higher_is_better_flipping(self):
        """Lower-is-better factors should flip z-scores."""
        from factors.quality import LowLeverageFactor
        llf = LowLeverageFactor()

        symbols = [f"S{i}" for i in range(20)]
        fund = {}
        for i, sym in enumerate(symbols):
            fund[sym] = {"Debt to equity": float(i)}  # S0 = 0 debt, S19 = 19 debt

        scores = llf.score({}, fund)
        # S0 (lowest debt) should have highest z-score
        assert scores["S0"].z_score > scores["S19"].z_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
