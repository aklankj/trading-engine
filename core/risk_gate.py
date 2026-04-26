"""
Risk gate — the final check before any order is placed.

Enforces hard limits that CANNOT be overridden by signals:
- Max position size per stock
- Max daily loss (circuit breaker)
- Max open positions
- Max sector concentration
- ATR-based stop-loss calculation
- Capital allocation between fundamental/active/cash
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date

from config.settings import cfg
from core.meta_allocator import CompositeSignal
from utils.logger import log
from utils import load_json, save_json, today_str, fmt_inr


@dataclass
class RiskDecision:
    """Output of the risk gate."""
    approved: bool
    reason: str
    position_size: int         # Number of shares
    position_value: float      # Total value in ₹
    stop_loss: float           # Stop-loss price
    target: float              # Target price
    risk_reward: float         # Risk/reward ratio
    pct_of_capital: float      # % of total capital this trade uses


@dataclass
class PortfolioState:
    """Current portfolio state for risk checks."""
    total_capital: float
    deployed_capital: float
    available_cash: float
    open_positions: int
    daily_pnl: float
    sector_exposure: dict[str, float]   # sector -> % of capital
    fundamental_allocation: float       # % deployed in CNC
    active_allocation: float            # % deployed in MIS/NRML


def get_portfolio_state() -> PortfolioState:
    """Build current portfolio state from trade log and positions."""
    trade_log = load_json(cfg.TRADE_LOG, default={"trades": [], "daily_pnl": {}})

    today = today_str()
    daily_pnl = trade_log.get("daily_pnl", {}).get(today, 0.0)

    # Count open positions from today's trades
    open_trades = [
        t for t in trade_log.get("trades", [])
        if t.get("status") == "open"
    ]

    sector_exposure = {}
    deployed = 0.0
    fundamental_deployed = 0.0
    active_deployed = 0.0

    for t in open_trades:
        val = t.get("value", 0)
        sector = t.get("sector", "Unknown")
        deployed += val
        sector_exposure[sector] = sector_exposure.get(sector, 0) + val
        if t.get("product") == "CNC":
            fundamental_deployed += val
        else:
            active_deployed += val

    total = cfg.INITIAL_CAPITAL + daily_pnl
    for s in sector_exposure:
        sector_exposure[s] = sector_exposure[s] / total if total > 0 else 0

    return PortfolioState(
        total_capital=total,
        deployed_capital=deployed,
        available_cash=total - deployed,
        open_positions=len(open_trades),
        daily_pnl=daily_pnl,
        sector_exposure=sector_exposure,
        fundamental_allocation=fundamental_deployed / total if total > 0 else 0,
        active_allocation=active_deployed / total if total > 0 else 0,
    )


def evaluate_risk(
    signal: CompositeSignal,
    symbol: str,
    sector: str,
    current_price: float,
    atr: float,
    product: str = "MIS",  # MIS (intraday), NRML (overnight), CNC (delivery)
) -> RiskDecision:
    """
    Evaluate whether a signal passes the risk gate.

    Args:
        signal: CompositeSignal from meta-allocator
        symbol: Trading symbol (e.g., "RELIANCE")
        sector: Sector for concentration check
        current_price: Current market price
        atr: Average True Range (14-period) for stop-loss
        product: Kite product type
    """
    state = get_portfolio_state()
    reasons = []

    # ── Circuit breaker: daily loss limit ─────────────────────
    daily_loss_pct = abs(state.daily_pnl) / state.total_capital if state.total_capital > 0 else 0
    if state.daily_pnl < 0 and daily_loss_pct >= cfg.MAX_DAILY_LOSS_PCT:
        return RiskDecision(
            approved=False,
            reason=f"CIRCUIT BREAKER: Daily loss {daily_loss_pct:.1%} exceeds {cfg.MAX_DAILY_LOSS_PCT:.0%} limit",
            position_size=0, position_value=0, stop_loss=0, target=0,
            risk_reward=0, pct_of_capital=0,
        )

    # ── Max open positions ────────────────────────────────────
    if state.open_positions >= cfg.MAX_OPEN_POSITIONS:
        return RiskDecision(
            approved=False,
            reason=f"Max open positions ({cfg.MAX_OPEN_POSITIONS}) reached",
            position_size=0, position_value=0, stop_loss=0, target=0,
            risk_reward=0, pct_of_capital=0,
        )

    # ── Sector concentration ──────────────────────────────────
    sector_pct = state.sector_exposure.get(sector, 0)
    if sector_pct >= cfg.MAX_SECTOR_EXPOSURE_PCT:
        reasons.append(f"Sector {sector} at {sector_pct:.0%} (max {cfg.MAX_SECTOR_EXPOSURE_PCT:.0%})")

    # ── Capital allocation check ──────────────────────────────
    if product == "CNC":
        if state.fundamental_allocation >= cfg.FUNDAMENTAL_ALLOC:
            reasons.append(f"Fundamental allocation full ({state.fundamental_allocation:.0%})")
    else:
        if state.active_allocation >= cfg.ACTIVE_ALLOC:
            reasons.append(f"Active allocation full ({state.active_allocation:.0%})")

    if reasons:
        return RiskDecision(
            approved=False, reason="; ".join(reasons),
            position_size=0, position_value=0, stop_loss=0, target=0,
            risk_reward=0, pct_of_capital=0,
        )

    # ── Position sizing (ATR-based) ───────────────────────────
    max_position_value = state.total_capital * cfg.MAX_POSITION_PCT
    risk_per_share = atr * 2  # 2x ATR stop-loss distance

    # Risk 1% of capital per trade
    risk_amount = state.total_capital * 0.01
    shares_by_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    shares_by_value = int(max_position_value / current_price) if current_price > 0 else 0

    position_size = min(shares_by_risk, shares_by_value)
    # Finding #13: If sizing math says 0, reject the trade — don't force min 1
    if position_size <= 0:
        return RiskDecision(
            approved=False, reason=f"Position size is zero (risk={shares_by_risk}, value={shares_by_value})",
            position_size=0, position_value=0, stop_loss=0, target=0,
            risk_reward=0, product=product
        )
    position_value = position_size * current_price

    # ── Stop-loss and target ──────────────────────────────────
    if signal.direction == "BUY":
        stop_loss = current_price - (atr * 2)
        target = current_price + (atr * 3)
    else:  # SELL
        stop_loss = current_price + (atr * 2)
        target = current_price - (atr * 3)

    risk_per_unit = abs(current_price - stop_loss)
    reward_per_unit = abs(target - current_price)
    risk_reward = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

    pct_of_capital = position_value / state.total_capital if state.total_capital > 0 else 0

    # ── Final approval ────────────────────────────────────────
    approved = True
    reason = "All risk checks passed"

    if pct_of_capital > cfg.MAX_POSITION_PCT:
        approved = False
        reason = f"Position size {pct_of_capital:.1%} exceeds {cfg.MAX_POSITION_PCT:.0%}"

    if risk_reward < 1.0:
        approved = False
        reason = f"Risk/reward {risk_reward:.2f} below 1.0 minimum"

    if signal.strength == "weak" and signal.agreement_pct < 0.5:
        approved = False
        reason = f"Weak signal ({signal.signal:+.2f}) with low agreement ({signal.agreement_pct:.0%})"

    result = RiskDecision(
        approved=approved,
        reason=reason,
        position_size=position_size,
        position_value=position_value,
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        risk_reward=round(risk_reward, 2),
        pct_of_capital=pct_of_capital,
    )

    log.info(
        f"Risk gate for {symbol}: {'✅ APPROVED' if approved else '❌ REJECTED'} | "
        f"Reason: {reason} | Size: {position_size} shares ({fmt_inr(position_value)}) | "
        f"SL: {stop_loss:.2f} Target: {target:.2f} R:R={risk_reward:.2f}"
    )
    return result
