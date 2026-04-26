import numpy as np
import pandas as pd
import pytest


def make_test_data(days=1000, start_price=100, trend=0.0005, volatility=0.02, seed=42):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=days, freq="B")
    prices = [start_price]
    for i in range(1, days):
        ret = trend + volatility * rng.standard_normal()
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "open": prices * (1 + 0.001 * rng.standard_normal(days)),
            "high": prices * (1 + np.abs(0.01 * rng.standard_normal(days))),
            "low": prices * (1 - np.abs(0.01 * rng.standard_normal(days))),
            "close": prices,
            "volume": rng.integers(100000, 1000000, days),
        },
        index=dates,
    )
    return df


@pytest.fixture
def synthetic_data():
    """Default synthetic data fixture (1000 rows)."""
    return make_test_data(days=1000)


@pytest.fixture
def tmp_portfolio_path(tmp_path):
    """Provide an isolated directory for portfolio JSON files."""
    return tmp_path
