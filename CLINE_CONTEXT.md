# Trading Engine — Cline Context

## Project Overview

Regime-adaptive algorithmic trading engine for Indian markets (NSE/MCX), deployed on a DigitalOcean VPS. Paper trading mode, targeting eventual live trading via Zerodha Kite Connect API.

**Owner:** Aklank Jain, PM at Juspay, Bangalore
**VPS:** 159.203.78.174 (Ubuntu 24.04, root access)
**Python:** 3.12 with venv at `~/trading-engine/venv`
**Broker:** Zerodha Kite Connect (API key: ju63hb9raw40c2n8, Client ID: ZI8488)
**Telegram Bot:** Rex (@Rex_aj95_bot), chat_id: 371697517
**LLM:** OpenRouter with google/gemini-2.0-flash-001
**Timezone:** Asia/Kolkata (IST)

---

## Repository Structure

```
trading-engine/
├── main.py                              # Orchestrator, CLI, scheduler
├── config/
│   ├── settings.py                      # Typed config, paths, constants
│   └── .env                             # Credentials (NEVER commit)
│
├── strategies/                          # ★ UNIFIED STRATEGY LAYER (V6)
│   ├── __init__.py
│   ├── base.py                          # BaseStrategy ABC + BacktestResult + metrics
│   ├── registry.py                      # Strategy registry, composite signal, weights
│   ├── quality_dip_buy.py               # Buy quality stocks on 15-30% dips (CAGR TBD)
│   ├── annual_momentum.py               # Buy if >10% 12m return + above 200 SMA
│   ├── weekly_trend.py                  # 20-week Donchian breakout
│   ├── donchian_monthly.py              # Monthly Donchian channel
│   ├── all_weather.py                   # Regime-adaptive: momentum in bull, reversion in bear
│   └── weekly_daily.py                  # Weekly trend + daily RSI pullback entry
│
├── backtest/
│   ├── __init__.py
│   ├── unified.py                       # ★ Uses strategies.registry (SAME code as live)
│   ├── engine.py                        # OLD backtester — DEPRECATED, do not use
│   └── strategy_lab.py                  # Strategy experimentation (standalone functions)
│
├── core/                                # Market data, regime detection, auth
│   ├── auth.py                          # Kite authentication + token persistence
│   ├── auto_auth.py                     # Automated daily TOTP login
│   ├── data.py                          # Candle fetch, indicators, portfolio
│   ├── regime.py                        # HMM + rule-based regime detector
│   ├── watchlist.py                     # Tiered watchlist (T1: 58, T2: 100, T3: 9336)
│   ├── risk_gate.py                     # Position sizing, circuit breaker, risk limits
│   ├── meta_allocator.py                # OLD composite signal — DEPRECATED
│   ├── strategies.py                    # OLD 8 strategies — DEPRECATED (negative Sharpe)
│   ├── swing_strategies.py              # OLD swing strategies — REPLACED by strategies/
│   ├── exits.py                         # Exit logic (partially superseded by strategy exits)
│   ├── multi_market.py                  # Global markets via yfinance
│   ├── macro_signals.py                 # FII/DII, VIX, PCR (not wired in)
│   └── correlation_costs.py             # Correlation guard + cost model (not wired in)
│
├── execution/
│   ├── orders.py                        # Paper/approval/semi-auto order execution
│   ├── portfolio_tracker.py             # ★ Paper portfolio with real P&L tracking
│   └── telegram_bot.py                  # Signal delivery, alerts, daily recap
│
├── jobs/
│   ├── morning_scan_v2.py               # 08:45 AM — regime scan, 148 instruments
│   ├── swing_scanner_v2.py              # ★ 10:00 AM — ONLY signal source (uses registry)
│   ├── swing_scanner.py                 # OLD — replaced by v2
│   ├── intraday_loop_v2.py              # OLD — killed in V6 (was running old strategies)
│   ├── evening_recap.py                 # 04:00 PM — P&L recap via portfolio_tracker
│   └── weekly_research.py               # Sunday 20:00 — arXiv paper discovery
│
├── fundamental/
│   ├── screener_scraper.py              # Screener.in company page scraper
│   ├── full_universe_scan.py            # All-NSE scan with resume support
│   ├── bear_opportunity.py              # Bear market opportunity scanner
│   ├── deep_analyzer.py                 # AI analysis of top picks via OpenRouter
│   ├── buy_triggers.py                  # Price-drop detection on quality stocks
│   └── macro_overlay.py                 # Sector/valuation adjustments
│
├── research/
│   ├── scanner.py                       # arXiv API paper discovery
│   └── strategy_pipeline.py             # Paper → code → backtest → promote/reject
│
├── data/                                # Runtime data (JSON files, logs, state)
│   ├── paper_portfolio.json             # Current paper positions + P&L
│   ├── paper_trade_history.json         # Closed trade records
│   ├── backtest_results.json            # Latest backtest output
│   ├── tiered_watchlist.json            # Built from Kite instruments
│   ├── fundamental_scan.json            # Screener.in results
│   ├── bear_opportunities.json          # Bear market picks
│   ├── regime_state.json                # Per-stock regime tracking
│   ├── strategy_pipeline.json           # Pipeline status per paper
│   └── generated_strategies/            # LLM-generated strategy code
│
├── deploy_v6.py                         # V6 deployment patch
├── critical_fixes.py                    # Adversarial review fixes
└── logs/                                # Daily engine logs
```

---

## Current State (April 2026)

### What works
- Auto Kite login at 6:30 AM IST via TOTP
- Morning regime scan of 148 instruments at 8:45 AM
- Swing signal scan at 10:00 AM using 6 backtested strategies
- Paper portfolio tracking with real entry/exit/P&L
- Evening recap with actual returns at 4:00 PM
- Weekly research paper discovery from arXiv
- Research → backtest pipeline (auto-tests LLM-generated strategies)
- Bear market opportunity scanner
- Full NSE universe fundamental scan (1,636 companies)

### What's broken or needs work
1. **V6 not fully deployed** — deploy_v6.py and critical_fixes.py need to be run
2. **Old engine still interfering** — `meta_composite` and old intraday loop generate bad trades alongside swing scanner
3. **Backtest metrics were wrong** — now fixed with gated Sharpe (n≥30), CAGR, expectancy
4. **114-stock backtest running** — results pending, will determine true strategy performance
5. **Walk-forward validation not yet run** — needed to confirm strategies aren't overfit
6. **No automated test suite** — adversarial review demanded regression tests
7. **No validation dashboard** — need daily tracking of equity, drawdown, strategy attribution

### Live paper trading results (2 weeks)
- 21 closed trades: 7 winners, 14 losers (33% win rate)
- Realized P&L: ₹+1,165 on ₹1L capital
- Biggest winner: Siemens +21.4% (+₹1,355)
- Biggest loser: BPCL -9.4% (-₹464)
- 8 of 14 losers came from the OLD dead engine (meta_composite)

---

## Architecture Principles

### Strategy Layer (strategies/)
- **Every strategy inherits from `BaseStrategy`** in `strategies/base.py`
- **Two abstract methods:** `should_enter(df, i)` and `should_exit(df, i, position)`
- **Same code runs in backtest AND live** — `strategy.backtest(df)` and `strategy.signal(df)` use the same `should_enter`/`should_exit`
- **Registry** (`strategies/registry.py`) is the ONLY source of active strategies
- **Weights** are based on backtested CAGR (not Sharpe — Sharpe requires n≥30)

### Backtest Layer (backtest/)
- **ONLY use `backtest/unified.py`** — it imports from `strategies.registry`
- `backtest/engine.py` is DEPRECATED (uses old strategies)
- `backtest/strategy_lab.py` is for experimentation only (standalone functions)
- Metrics: CAGR (primary), Expectancy, Profit Factor, Max Drawdown, Sharpe (gated)
- Transaction costs: 0.2% round-trip per trade (STT + brokerage + slippage)

### Execution Layer
- **`swing_scanner_v2.py` is the ONLY signal source** — nothing else opens positions
- **`portfolio_tracker.py`** tracks all paper positions with real entry prices, SL/TGT exits
- Exit logic comes from strategy's `max_hold_days`, NOT a global 14-day timer
- SELL signals rejected for cash equities (not shortable in India)
- All numeric fields cast with `_safe_float` / `_safe_int` to prevent type crashes

### Data Flow
```
Kite API → fetch_candles() → strategies.registry.get_composite_signal() → portfolio_tracker.open_position()
                                                                        → portfolio_tracker.update_positions() (check SL/TGT)
                                                                        → portfolio_tracker.close_position() (record P&L)
                                                                        → evening_recap.run() → Telegram
```

---

## 6 Core Strategies

| Strategy | What it does | Hold period | Key indicator |
|---|---|---|---|
| QualityDipBuy | Buy quality stocks down 15-30% from 52w high, above 200 SMA | 30-90 days | Drop from high + SMA200 |
| AnnualMomentum | Buy if >10% 12m return, above 200 SMA, rebalance monthly | 12 months | 12m return + SMA200 |
| WeeklyTrend | Enter on 20-week Donchian breakout, exit on 10-week low | 4-26 weeks | Weekly Donchian channel |
| DonchianMonthly | Monthly Donchian channel for big trends | 1-12 months | Weekly Donchian channel |
| AllWeather | Momentum pullbacks in bull, mean reversion in bear | 30-90 days | Regime + RSI + SMA |
| WeeklyDaily | Weekly uptrend + daily RSI<40 pullback entry | 15-45 days | Weekly SMA + daily RSI |

---

## Deployment Commands

```bash
# SSH into VPS
ssh root@159.203.78.174
cd ~/trading-engine && source venv/bin/activate

# Deploy new code
scp <file>.tar.gz root@159.203.78.174:~/
tar -xzf ~/<file>.tar.gz --strip-components=1

# Apply patches
python critical_fixes.py
python deploy_v6.py

# Reset paper portfolio (only when starting fresh)
python -c "from execution.portfolio_tracker import reset_portfolio; reset_portfolio()"

# Run backtests
python -m backtest.unified                         # 10 stocks, quick check
python -m backtest.unified --full                   # 47 stocks (NIFTY 50)
python -m backtest.unified --nifty100               # 114 stocks
python -m backtest.unified --nifty100 --walkforward # Out-of-sample validation
python -m backtest.unified --years=15               # 15 years
python -m backtest.unified --symbol=RELIANCE        # Single stock

# Test live signals
python main.py --swingscan

# Start engine
pkill -f "python main.py"
nohup python main.py > engine_output.log 2>&1 &
echo $! > engine.pid

# Check status
python main.py --portfolio         # Paper portfolio P&L
tail -20 engine_output.log         # Engine health
tail -20 backtest_n100.log         # Backtest progress
```

---

## Critical Rules for Cline

### DO:
- Always use `strategies/base.py` BaseStrategy for new strategies
- Always use `.iloc[i]` for positional DataFrame indexing (NEVER `df['col'][i]`)
- Cast all portfolio numeric fields with `float()` / `int()` before arithmetic
- Run backtest on the SAME code that runs live
- Gate Sharpe ratio at n≥30 trades minimum
- Deduct transaction costs in backtests (0.2% round-trip)
- Reject SELL signals for cash equities (India: no shorting without F&O)
- Test on 100+ stocks to avoid survivorship bias

### DO NOT:
- Import from `core/strategies.py` — it's DEAD (negative Sharpe, killed)
- Import from `core/swing_strategies.py` — it's DEAD (replaced by `strategies/`)
- Import from `core/meta_allocator.py` — it's DEAD
- Use `backtest/engine.py` — it's DEPRECATED (uses old strategies)
- Execute raw LLM code with `exec()` without sandboxing
- Force minimum 1 share when sizing says 0 (reject the trade instead)
- Trust Sharpe ratios on fewer than 30 trades
- Store credentials in code (use .env)
- Open the same position twice (check existing positions first)

### NEVER:
- Place real orders without explicit user approval
- Run both old intraday loop AND swing scanner (causes duplicate trades)
- Use hardcoded 14-day position exit (use strategy's max_hold_days)
- Trust scraped data without validation (Screener.in HTML can change)

---

## Pending Work (Priority Order)

### P0 — Deploy V6 + Critical Fixes
- [ ] Run `critical_fixes.py` on VPS
- [ ] Run `deploy_v6.py` on VPS  
- [ ] Verify old intraday loop is completely dead
- [ ] Reset portfolio, start fresh tracking

### P1 — Validate Strategies
- [ ] Review 114-stock backtest results (running now)
- [ ] Run walk-forward validation (70% train / 30% test)
- [ ] If strategies fail walk-forward, downweight or remove them
- [ ] Determine final strategy weights based on CAGR, not old Sharpe claims

### P2 — Build Test Suite
- [ ] `tests/test_strategy_base.py` — backtest produces correct results
- [ ] `tests/test_portfolio_tracker.py` — open/close/P&L tracking
- [ ] `tests/test_swing_scanner.py` — SELL rejected for cash equities, SL/TGT never zero
- [ ] `tests/test_risk_gate.py` — zero size rejects trade
- [ ] `tests/test_metrics.py` — Sharpe gated at n<30, CAGR correct

### P3 — Build Validation Dashboard
- [ ] Daily equity curve tracking
- [ ] Per-strategy attribution (which strategy made/lost money)
- [ ] Drawdown monitoring with circuit breaker
- [ ] Missed job detection (did morning scan / swing scan / recap actually fire?)
- [ ] Crash detection and alerting

### P4 — Operational Hardening
- [ ] systemd service (replace nohup)
- [ ] Health check endpoint
- [ ] Portfolio persistence survives restarts
- [ ] Auth failure explicit alerting
- [ ] Cooldown state persisted to disk

### P5 — Enhanced Backtesting
- [ ] Monte Carlo simulation (randomized fills/ordering)
- [ ] Regime-segmented performance (bull/bear/sideways/panic results)
- [ ] Parameter sensitivity analysis (small changes still profitable?)
- [ ] Improved cost model (separate STT, brokerage, slippage by liquidity)

---

## Adversarial Review Summary (23 findings)

### CRITICAL (fixed or in progress):
1. Live ≠ backtest code → Fixed by unified strategies/
2. Stale backtest results → Fixed by unified backtester
3. 14-day forced exit → Fixed in critical_fixes.py
4. Type crash in portfolio → Fixed with _safe_float
5. Unsandboxed LLM exec → Partially fixed (restricted builtins)
6. Brittle auth → Monitor for now

### HIGH (partially addressed):
7. Backtest P&L calc → Fixed with proper equity curve + costs
8. No transaction costs → 0.2% round-trip added
9. SELL on non-shortable → Rejected in swing_scanner_v2
10. No mark-to-market risk → TODO: portfolio-level circuit breaker
11-12. Schedule/loop issues → Killed in V6
13. Forced min 1 share → Fixed: zero = reject
14. MCX expiry string sort → Fixed: date parsing
15. Correlation/cost not wired → TODO

### MEDIUM/LOW:
16-23. Auth handling, pipeline dedup, cooldown persistence, Telegram escaping, config validation

---

## Key File Relationships

```
strategies/registry.py  ──imports──> strategies/base.py
                        ──imports──> strategies/quality_dip_buy.py (etc.)
                        
backtest/unified.py     ──imports──> strategies/registry.py  (SAME strategies)

jobs/swing_scanner_v2.py ──imports──> strategies/registry.py  (SAME strategies)
                         ──imports──> execution/portfolio_tracker.py

jobs/evening_recap.py    ──imports──> execution/portfolio_tracker.py

main.py                 ──schedules──> jobs/morning_scan_v2.py (08:45)
                        ──schedules──> jobs/swing_scanner_v2.py (10:00)
                        ──schedules──> jobs/evening_recap.py (16:00)
                        ──schedules──> jobs/weekly_research.py (Sunday 20:00)
```

---

## VPS Environment

```bash
# Python packages installed
kiteconnect yfinance hmmlearn pandas numpy scipy scikit-learn
requests beautifulsoup4 feedparser pyotp schedule loguru python-dotenv

# Network: egress restricted (may need proxy for some APIs)
# Disk: JSON file storage in data/ (no database)
# Memory: 2GB RAM (sufficient for current workload)
# Process: single-threaded scheduler via `schedule` library
```
