# Sprint: Test Suite Build

## Goal
Build regression tests that prove the trading engine works correctly.
Every critical fix from the adversarial review must have a corresponding test.
Tests run locally (no VPS, no Kite API, no network needed).

## Setup

```bash
cd ~/Documents/trading-engine
pip install pytest pytest-cov
mkdir -p tests
touch tests/__init__.py
```

Run all tests: `pytest tests/ -v`
Run single file: `pytest tests/test_portfolio_tracker.py -v`
Run with coverage: `pytest tests/ --cov=strategies --cov=execution -v`

---

## Task 1: test_strategy_base.py
**What:** Verify BaseStrategy backtest mechanics work correctly
**Why:** Finding #1 — backtest must use same code as live
**File:** `tests/test_strategy_base.py`

### Tests to write:

```
test_backtest_produces_trades
  - Create a simple strategy that always buys
  - Run backtest on synthetic data (pd.DataFrame with 500 rows)
  - Assert: total_trades > 0

test_backtest_and_signal_use_same_enter_logic
  - Create a strategy, run backtest, also call signal() on last bar
  - Assert: both call should_enter with same indicators

test_sharpe_gated_below_30_trades
  - Run backtest that produces 10 trades
  - Assert: result.sharpe_valid == False
  - Assert: result.sharpe is still computed (just marked invalid)

test_sharpe_valid_above_30_trades
  - Run backtest that produces 40+ trades
  - Assert: result.sharpe_valid == True

test_cagr_calculated_correctly
  - Synthetic backtest: 100K → 200K over 5 years
  - Assert: CAGR ≈ 14.87% (compound growth formula)

test_expectancy_calculated_correctly
  - 10 trades: 6 wins at +5%, 4 losses at -3%
  - Assert: expectancy = 0.6*5 + 0.4*(-3) = 1.8%

test_transaction_costs_reduce_equity
  - Run same backtest with and without costs
  - Assert: equity_with_costs < equity_without_costs

test_max_drawdown_calculated
  - Synthetic trades: +10%, -20%, +5%
  - Assert: max_drawdown > 0 and reasonable

test_empty_dataframe_returns_empty_result
  - Pass empty DataFrame
  - Assert: total_trades == 0, no crash

test_insufficient_data_returns_empty_result
  - Pass 50 rows (below min_bars)
  - Assert: total_trades == 0, no crash
```

### Synthetic data helper:
```python
def make_test_data(days=1000, start_price=100, trend=0.0005, volatility=0.02):
    """Generate synthetic OHLCV data for testing."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range('2016-01-01', periods=days, freq='B')
    prices = [start_price]
    for i in range(1, days):
        ret = trend + volatility * np.random.randn()
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)
    df = pd.DataFrame({
        'open': prices * (1 + 0.001 * np.random.randn(days)),
        'high': prices * (1 + abs(0.01 * np.random.randn(days))),
        'low': prices * (1 - abs(0.01 * np.random.randn(days))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days),
    }, index=dates)
    return df
```

### Acceptance criteria:
- [ ] All 10 tests pass
- [ ] No imports from core/strategies.py or core/meta_allocator.py
- [ ] Tests run without network (no yfinance, no Kite)

---

## Task 2: test_portfolio_tracker.py
**What:** Verify portfolio open/close/P&L lifecycle
**Why:** Findings #3, #4 — type bugs and forced exits broke P&L tracking
**File:** `tests/test_portfolio_tracker.py`

### Tests to write:

```
test_open_position_deducts_cash
  - Reset portfolio, open position
  - Assert: cash decreased by position value

test_open_position_prevents_duplicate
  - Open RELIANCE, try opening RELIANCE again
  - Assert: still 1 position, not 2

test_close_position_records_pnl
  - Open BUY position at 100, close at 120
  - Assert: realized_pnl = +20 * quantity
  - Assert: position removed from portfolio
  - Assert: cash increased

test_stop_loss_triggers_on_sell
  - Open SELL at 1000, SL=1050
  - Update with price=1060
  - Assert: position closed with reason "stop_loss"
  - Assert: P&L is negative

test_target_triggers_on_buy
  - Open BUY at 100, TGT=120
  - Update with price=125
  - Assert: position closed with reason "target_hit"
  - Assert: P&L is positive

test_max_hold_days_respected
  - Open position with max_hold_days=90
  - Simulate 91 days
  - Assert: position closed with reason "time_exit"

test_no_14_day_forced_exit
  - Open position with max_hold_days=90
  - Simulate 15 days
  - Assert: position still OPEN (not force-closed)

test_numeric_type_safety
  - Open position, manually corrupt entry_price to string "100.5"
  - Call update_positions
  - Assert: no crash, values handled correctly

test_trade_history_recorded
  - Open and close a position
  - Assert: paper_trade_history.json has 1 entry
  - Assert: entry has all fields (symbol, pnl, exit_reason, strategy, etc.)

test_win_rate_tracked
  - Close 3 winners and 2 losers
  - Assert: win_rate = 60%
  - Assert: winning_trades = 3, losing_trades = 2

test_reset_clears_everything
  - Open positions, close some
  - Reset
  - Assert: cash=100000, positions={}, trades=0

test_portfolio_survives_restart
  - Open positions, save
  - Reload from disk
  - Assert: positions intact, cash correct
```

### Acceptance criteria:
- [ ] All 12 tests pass
- [ ] Tests use tmp_path fixture (no writing to real data/)
- [ ] No network calls

---

## Task 3: test_strategies.py
**What:** Verify each of the 6 strategies produces valid signals
**Why:** Finding #2 — strategies must be individually validated
**File:** `tests/test_strategies.py`

### Tests to write:

```
test_quality_dip_buy_signal_on_dip
  - Create data where stock dropped 20% from high but above 200 SMA
  - Assert: signal.direction == "BUY"
  - Assert: signal.stop_loss > 0
  - Assert: signal.target > signal.stop_loss

test_quality_dip_buy_no_signal_small_dip
  - Create data where stock dropped only 5%
  - Assert: signal.direction == "HOLD"

test_quality_dip_buy_no_signal_below_sma
  - Create data where stock dropped 25% AND below 200 SMA
  - Assert: signal.direction == "HOLD"

test_annual_momentum_buy_on_positive_year
  - Create uptrending data (+15% over 252 bars), above 200 SMA
  - Assert: signal.direction == "BUY" (or HOLD with low confidence if not month start)

test_annual_momentum_no_signal_negative_year
  - Create downtrending data (-10% over 252 bars)
  - Assert: signal.direction != "BUY"

test_weekly_trend_buy_on_breakout
  - Create data that breaks 20-week high
  - Assert: signal.direction == "BUY"
  - Assert: signal.stop_loss > 0 (NEVER zero)

test_donchian_monthly_sell_has_valid_sl_tgt
  - Create data that breaks 10-week low
  - Assert: signal.stop_loss > 0 (this was the zero-SL bug)
  - Assert: signal.target > 0

test_all_weather_adapts_to_regime
  - Test bull data: assert BUY signal on pullback
  - Test bear data: assert BUY only on extreme oversold
  - Test sideways data: assert HOLD

test_weekly_daily_requires_weekly_uptrend
  - Create data with weekly downtrend + daily RSI < 40
  - Assert: signal.direction == "HOLD" (weekly filter rejects)

test_all_strategies_handle_short_data
  - Pass 50 rows to each strategy
  - Assert: all return HOLD, no crashes

test_all_strategies_return_valid_signal_type
  - Run each strategy on synthetic data
  - Assert: returns SwingSignal with valid fields
```

### Acceptance criteria:
- [ ] All 11 tests pass
- [ ] Each strategy tested in isolation
- [ ] No zero SL/TGT on any signal with direction != HOLD

---

## Task 4: test_risk_gate.py
**What:** Verify position sizing and trade rejection
**Why:** Finding #13 — zero size must reject, not force min 1
**File:** `tests/test_risk_gate.py`

### Tests to write:

```
test_zero_size_rejected
  - Call evaluate_risk with very small capital / very expensive stock
  - Assert: approved == False
  - Assert: position_size == 0

test_position_size_within_limit
  - Call evaluate_risk with normal params
  - Assert: position_value / capital <= 0.05 (5% limit)

test_risk_reward_calculated
  - Provide entry=100, SL=90, TGT=120
  - Assert: risk_reward = 2.0 (20/10)
```

### Acceptance criteria:
- [ ] All 3 tests pass
- [ ] No forced min(1, ...) in sizing

---

## Task 5: test_swing_scanner.py
**What:** Verify signal flow from scanner to portfolio
**Why:** Finding #9 — SELL must be rejected for cash equities
**File:** `tests/test_swing_scanner.py`

### Tests to write:

```
test_sell_rejected_for_cash_equity
  - Mock a SELL signal for an equity stock
  - Assert: position NOT opened

test_zero_sl_tgt_rejected
  - Mock a signal with SL=0
  - Assert: position NOT opened

test_max_positions_enforced
  - Open 10 positions
  - Try to open 11th
  - Assert: rejected, still 10

test_max_new_per_day_enforced
  - Open 3 positions in one scan
  - Assert: 4th signal skipped even if valid
```

### Acceptance criteria:
- [ ] All 4 tests pass
- [ ] Tests mock Kite API (no real network calls)

---

## Task 6: test_metrics.py
**What:** Verify metric calculations are mathematically correct
**Why:** Sharpe was inflated, CAGR was missing
**File:** `tests/test_metrics.py`

### Tests to write:

```
test_cagr_formula
  - 100K → 200K in 5 years → CAGR = 14.87%
  - 100K → 100K in 10 years → CAGR = 0%
  - 100K → 50K in 3 years → CAGR = -20.6%

test_expectancy_formula
  - Known trades: [+10, +5, -3, -4, +8]
  - Assert: expectancy = mean = 3.2%

test_profit_factor
  - Wins: [10, 5, 8], Losses: [-3, -4]
  - Assert: PF = 23/7 = 3.29

test_sharpe_gating
  - 5 trades → sharpe_valid = False
  - 30 trades → sharpe_valid = True
  - 0 trades → sharpe = 0

test_max_drawdown
  - Equity curve: 100, 110, 90, 95, 80, 100
  - Assert: max_dd = (110-80)/110 = 27.3%
```

### Acceptance criteria:
- [ ] All 5 tests pass
- [ ] Hand-calculated expected values

---

## Sprint Definition of Done

- [ ] 45 tests total across 6 files
- [ ] All pass: `pytest tests/ -v` shows 45 passed, 0 failed
- [ ] Coverage: `pytest tests/ --cov=strategies --cov=execution` shows >80%
- [ ] No test requires network, VPS, Kite API, or real data files
- [ ] Tests use synthetic data or mocks
- [ ] conftest.py provides shared fixtures (make_test_data, tmp portfolio path)

---

## Cline Execution Order

1. Create `tests/conftest.py` with `make_test_data` fixture and tmp_path portfolio setup
2. Task 1: `test_strategy_base.py` (most foundational)
3. Task 6: `test_metrics.py` (validates the numbers everything depends on)
4. Task 2: `test_portfolio_tracker.py` (validates P&L tracking)
5. Task 3: `test_strategies.py` (validates each strategy individually)
6. Task 4: `test_risk_gate.py` (validates sizing)
7. Task 5: `test_swing_scanner.py` (validates signal flow)

After all tests pass locally, deploy V6 to VPS and run `pytest` there too.
