# Factor Engine

## What This Is

A cross-sectional multi-factor ranking engine for Indian equities. Instead of asking "should I buy this stock now?" (timing), it asks "which stocks have the highest expected returns over the next 1-3 months?" (ranking).

This is the single most robust way to generate alpha in equity markets. The academic evidence spans 30+ years across 40+ countries, including India.

## Quick Start

```bash
cd ~/trading-engine
source venv/bin/activate

# Install dependency (if not already)
pip install scipy --break-system-packages

# Quick test on NIFTY 50 (fastest, ~2 min)
python run_factor_backtest.py --universe nifty50

# Full backtest on NIFTY 200 (~10 min download, ~5 min backtest)
python run_factor_backtest.py --universe nifty200

# Try different presets
python run_factor_backtest.py --preset pure_momentum --universe nifty100
python run_factor_backtest.py --preset quality_first --universe nifty200
python run_factor_backtest.py --preset vol_adjusted --top-n 30

# Monthly rebalance (paper trading, dry run)
python -m factors.rebalancer

# Monthly rebalance (execute paper trades)
python -m factors.rebalancer --live
```

## Architecture

```
factors/
├── __init__.py         # Package exports
├── base.py             # BaseFactor ABC, FactorScore, CompositeScore dataclasses
├── momentum.py         # 12-1 Momentum, Short-Term Reversal, Vol-Adjusted Momentum
├── quality.py          # Quality composite (ROE + ROCE + low debt + growth + margin)
├── value.py            # Value composite (earnings yield + book-to-price)
├── composite.py        # Multi-factor engine: rank → filter → select → weight
├── universe.py         # NIFTY 200 universe, yfinance batch fetcher, sector map
├── backtest.py         # Cross-sectional factor backtest (monthly rebalance)
└── rebalancer.py       # Live monthly rebalance job (paper or Kite)
```

## How It Works

### 1. Factor Scoring
Each factor computes a raw value for every stock (e.g., 12-month return for momentum). These raw values are then cross-sectionally z-scored — meaning we rank each stock relative to the entire universe, not against an absolute threshold.

### 2. Composite Ranking
Multiple factors are combined with weights:
- **Momentum (50%)**: 12-month return, skipping the most recent month
- **Quality (35%)**: ROE + ROCE + low debt/equity + profit growth + margins
- **Value (15%)**: Earnings yield + book-to-price

### 3. Portfolio Construction
- Select top 20 stocks by composite rank
- Apply sector concentration limits (max 30% per sector)
- Equal-weight or score-weighted
- Rebalance monthly on last trading day

### 4. Transaction Costs
- 0.2% round-trip (STT + brokerage + slippage)
- Applied on all buy/sell turnover at rebalance
- Turnover typically 30-50% per month

## Presets

| Preset | Factors | Expected Alpha | Risk |
|---|---|---|---|
| `momentum_quality` | 50% Mom + 35% Qual + 15% Val | 4-8% | Medium |
| `pure_momentum` | 100% Momentum | 6-10% | High (crash risk) |
| `quality_first` | Quality filter → Mom + Val | 3-6% | Low |
| `vol_adjusted` | Vol-adj Mom + Qual + Val | 4-7% | Medium-Low |

## Key Differences from strategies/

| | strategies/ (old) | factors/ (new) |
|---|---|---|
| Question | "Buy this stock now?" | "Which stocks rank highest?" |
| Signal | Binary (BUY/SELL) | Continuous (rank 1-200) |
| Holding period | Variable (days-months) | Fixed (monthly rebalance) |
| Universe | One stock at a time | All stocks simultaneously |
| Alpha source | Timing | Selection |
| Backtest CAGR | 0.4-2.2% | Target: 15-25% |
| Benchmark | N/A | NIFTY 50 / Equal-weight |

## Integration with Existing Engine

### Data Sources
- **Backtest**: yfinance batch download (same as `backtest/engine.py`)
- **Live**: Kite API via `core/data.py` (same auth, same tokens)
- **Fundamentals**: Screener.in via `fundamental/screener_scraper.py`

### Adding to Daily Schedule
```python
# In main.py, add to scheduler:
# Last trading day of month at 2:30 PM
schedule.every().day.at("14:30").do(check_monthly_rebalance)

def check_monthly_rebalance():
    """Run factor rebalance on last trading day of month."""
    from datetime import datetime
    import calendar
    today = datetime.now()
    last_day = calendar.monthrange(today.year, today.month)[1]
    # Run on last 2 business days of month (in case market closed on last)
    if today.day >= last_day - 2:
        from factors.rebalancer import run_monthly_rebalance
        summary = run_monthly_rebalance(dry_run=True)  # Start with dry run
```

### Coexistence with strategies/
Both systems can run simultaneously:
- `strategies/` generates swing trade signals (intraday alerts)
- `factors/` manages the core portfolio (monthly rebalance)
- Separate portfolio state files (`paper_portfolio.json` vs `factor_portfolio.json`)

## Adding New Factors

```python
from factors.base import BaseFactor

class MyFactor(BaseFactor):
    name = "MyFactor"
    description = "What it measures"
    lookback_days = 252  # Trading days needed
    higher_is_better = True  # Or False for things like debt

    def compute_raw(self, price_data, fundamental_data=None, as_of_date=None):
        scores = {}
        for sym, df in price_data.items():
            # Your logic here
            scores[sym] = some_value
        return scores
```

Then add to a preset in `composite.py`:
```python
from factors.composite import FactorConfig
config = FactorConfig(factor=MyFactor(), weight=0.20)
```

## What to Watch For

1. **Alpha > 3% CAGR over benchmark**: Real edge, worth paper trading
2. **Sharpe > 0.8**: Acceptable risk-adjusted returns
3. **Max drawdown < 25%**: Survivable in bad years
4. **Hit rate > 55%**: Beating benchmark most months
5. **Turnover < 50%/month**: Costs aren't eating alpha
6. **Consistent yearly returns**: Not driven by 1-2 lucky years

## Next Steps After Backtest

1. **Walk-forward validation**: Split data into 5yr train / 1yr test windows
2. **Parameter sensitivity**: Does top-15 vs top-20 vs top-30 matter?
3. **Regime filter**: Reduce exposure in bear markets (use your HMM)
4. **Paper trade 3 months**: Track real vs expected performance
5. **Go live**: Start with 50% of capital, scale up over 6 months
