"""
Paper Trade Portfolio Tracker.

Tracks simulated positions with real entry/exit prices.
Produces actual P&L instead of the meaningless ₹0.

Replaces the old "log and forget" paper trading approach.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import cfg
from utils.logger import log
from utils import load_json, save_json, now_ist
from utils.costs import transaction_cost, round_trip_cost

def _safe_float(val, default=0.0):
    """Safely convert any value to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_int(val, default=0):
    """Safely convert any value to int."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default




PORTFOLIO_FILE = cfg.DATA_DIR / "paper_portfolio.json"
TRADE_HISTORY_FILE = cfg.DATA_DIR / "paper_trade_history.json"


def _load_portfolio() -> dict:
    """Load current paper portfolio state."""
    default = {
        "capital": 100000,
        "cash": 100000,
        "positions": {},
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_realized_pnl": 0,
        "peak_equity": 100000,
        "max_drawdown": 0,
        "created_at": datetime.now().isoformat(),
    }
    return load_json(PORTFOLIO_FILE, default=default)


def _save_portfolio(portfolio: dict):
    """Save portfolio state."""
    save_json(PORTFOLIO_FILE, portfolio)


def _load_history() -> list:
    """Load trade history."""
    return load_json(TRADE_HISTORY_FILE, default=[])


def _save_history(history: list):
    """Save trade history."""
    save_json(TRADE_HISTORY_FILE, history)


def open_position(
    symbol: str,
    direction: str,
    quantity: int,
    entry_price: float,
    stop_loss: float,
    target: float,
    regime: str,
    signal_strength: float,
    strategy: str = "meta_composite",
) -> dict:
    """
    Open a new paper position. Records entry price for real P&L tracking.
    Returns the position dict.
    """
    portfolio = _load_portfolio()

    # Zero quantity guard
    if quantity <= 0:
        log.debug(f"Skipping {symbol} open — zero or negative quantity: {quantity}")
        return {}

    # Check if already have position in this symbol
    if symbol in portfolio["positions"]:
        log.debug(f"Already have position in {symbol}, skipping")
        return portfolio["positions"][symbol]

    # Calculate position value
    value = quantity * entry_price

    # Check if we have enough cash
    if value > portfolio["cash"]:
        log.debug(f"Not enough cash for {symbol} (need ₹{value:.0f}, have ₹{portfolio['cash']:.0f})")
        return {}

    position = {
        "symbol": symbol,
        "direction": direction,
        "quantity": quantity,
        "entry_price": entry_price,
        "entry_date": now_ist().isoformat(),
        "stop_loss": stop_loss,
        "target": target,
        "regime_at_entry": regime,
        "signal_strength": signal_strength,
        "strategy": strategy,
        "current_price": entry_price,
        "unrealized_pnl": 0,
        "unrealized_pnl_pct": 0,
        "max_favorable": 0,
        "max_adverse": 0,
        "days_held": 0,
    }

    portfolio["positions"][symbol] = position
    portfolio["cash"] -= value
    _save_portfolio(portfolio)

    log.info(
        f"📝 PAPER OPEN: {direction} {quantity} {symbol} @ ₹{entry_price:.2f} "
        f"(SL=₹{stop_loss:.2f}, TGT=₹{target:.2f})"
    )

    return position


def update_positions(price_data: dict):
    """
    Update all open positions with current prices.
    price_data: {symbol: current_price}

    Checks stop-loss and target exits.
    """
    portfolio = _load_portfolio()
    closed = []

    for symbol, pos in list(portfolio["positions"].items()):
        if symbol not in price_data:
            continue

        current = price_data[symbol]
        pos["current_price"] = current

        # Calculate unrealized P&L
        if pos["direction"] == "BUY":
            pnl = (current - float(pos["entry_price"])) * int(pos["quantity"])
            pnl_pct = (current - float(pos["entry_price"])) / float(pos["entry_price"]) * 100
        else:
            pnl = (float(pos["entry_price"]) - current) * int(pos["quantity"])
            pnl_pct = (float(pos["entry_price"]) - current) / float(pos["entry_price"]) * 100

        pos["unrealized_pnl"] = round(pnl, 2)
        pos["unrealized_pnl_pct"] = round(pnl_pct, 2)

        # Track max favorable/adverse excursion
        pos["max_favorable"] = max(pos.get("max_favorable", 0), pnl_pct)
        pos["max_adverse"] = min(pos.get("max_adverse", 0), pnl_pct)

        # Days held
        entry_dt = datetime.fromisoformat(pos["entry_date"])
        pos["days_held"] = (now_ist() - entry_dt).days

        # Check exit conditions
        exit_reason = None

        if pos["direction"] == "BUY":
            if current <= pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif current >= pos["target"]:
                exit_reason = "target_hit"
        else:  # SELL
            if current >= pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif current <= pos["target"]:
                exit_reason = "target_hit"

        # Time-based exit: use strategy's hold period, not a global timer
        max_hold = int(pos.get("max_hold_days", 90))  # Default 90 if not set
        if pos.get("days_held", 0) >= max_hold:
            exit_reason = "time_exit"

        if exit_reason:
            closed.append((symbol, exit_reason, current, pnl, pnl_pct))

    # Close positions that hit exits
    if closed:
        for symbol, reason, price, pnl, pnl_pct in closed:
            close_position(symbol, price, reason)
        # Reload after closes (close_position saves its own state)
        portfolio = _load_portfolio()
    else:
        _save_portfolio(portfolio)


def close_position(symbol: str, exit_price: float, reason: str = "manual") -> dict:
    """Close a paper position and record realized P&L."""
    portfolio = _load_portfolio()

    if symbol not in portfolio["positions"]:
        return {}

    pos = portfolio["positions"][symbol]

    # Safe numeric conversion for all arithmetic inputs
    entry_price = _safe_float(pos.get("entry_price", 0))
    quantity = _safe_int(pos.get("quantity", 0))
    exit_price_f = _safe_float(exit_price)

    # Calculate realized P&L
    if pos["direction"] == "BUY":
        pnl = (exit_price_f - entry_price) * quantity
        pnl_pct = (exit_price_f - entry_price) / entry_price * 100 if entry_price > 0 else 0
    else:
        pnl = (entry_price - exit_price_f) * quantity
        pnl_pct = (entry_price - exit_price_f) / entry_price * 100 if entry_price > 0 else 0

    # Deduct transaction costs (entry + exit) using the canonical cost model
    cost = transaction_cost(entry_price, quantity, "entry") + transaction_cost(exit_price_f, quantity, "exit")
    net_pnl = pnl - cost

    # Record trade
    trade_record = {
        "symbol": symbol,
        "direction": pos["direction"],
        "quantity": quantity,
        "entry_price": round(entry_price, 2),
        "entry_date": pos["entry_date"],
        "exit_price": round(exit_price_f, 2),
        "exit_date": now_ist().isoformat(),
        "exit_reason": reason,
        "pnl": round(net_pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "transaction_cost": round(cost, 2),
        "days_held": _safe_int(pos.get("days_held", 0)),
        "regime_at_entry": pos.get("regime_at_entry", "Unknown"),
        "signal_strength": pos.get("signal_strength", 0),
        "strategy": pos.get("strategy", "unknown"),
        "max_favorable_pct": round(pos.get("max_favorable", 0), 2),
        "max_adverse_pct": round(pos.get("max_adverse", 0), 2),
    }

    history = _load_history()
    history.append(trade_record)
    _save_history(history)

    # Update portfolio stats using net P&L (after transaction costs)
    value = quantity * exit_price_f
    portfolio["cash"] += value
    portfolio["total_realized_pnl"] += net_pnl
    portfolio["total_trades"] += 1

    if net_pnl > 0:
        portfolio["winning_trades"] += 1
    else:
        portfolio["losing_trades"] += 1

    # Track equity and drawdown
    equity = get_total_equity(portfolio)
    portfolio["peak_equity"] = max(portfolio.get("peak_equity", 100000), equity)
    if portfolio["peak_equity"] > 0:
        dd = (portfolio["peak_equity"] - equity) / portfolio["peak_equity"] * 100
        portfolio["max_drawdown"] = max(portfolio.get("max_drawdown", 0), dd)

    del portfolio["positions"][symbol]
    _save_portfolio(portfolio)

    log.info(
        f"📝 PAPER CLOSE: {pos['direction']} {quantity} {symbol} "
        f"@ ₹{exit_price_f:.2f} | P&L: ₹{net_pnl:+.0f} ({pnl_pct:+.1f}%) | "
        f"Cost: ₹{cost:.2f} | Reason: {reason}"
    )

    return trade_record


def get_total_equity(portfolio: dict = None) -> float:
    """Calculate total equity (cash + positions)."""
    if portfolio is None:
        portfolio = _load_portfolio()

    equity = portfolio["cash"]
    for pos in portfolio["positions"].values():
        equity += float(pos["quantity"]) * float(pos.get("current_price", pos["entry_price"]))

    return equity


def get_daily_summary() -> dict:
    """Generate daily P&L summary for the evening recap."""
    portfolio = _load_portfolio()
    history = _load_history()
    equity = get_total_equity(portfolio)

    # Today's closed trades
    today = now_ist().strftime("%Y-%m-%d")
    today_trades = [t for t in history if t["exit_date"][:10] == today]
    today_pnl = sum(t["pnl"] for t in today_trades)

    # Overall stats
    total_pnl = portfolio.get("total_realized_pnl", 0)
    unrealized = sum(p.get("unrealized_pnl", 0) for p in portfolio["positions"].values())
    total_trades = portfolio.get("total_trades", 0)
    wins = portfolio.get("winning_trades", 0)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # Per-strategy breakdown
    strategy_pnl = {}
    for t in history:
        strat = t.get("strategy", "unknown")
        if strat not in strategy_pnl:
            strategy_pnl[strat] = {"pnl": 0, "trades": 0, "wins": 0}
        strategy_pnl[strat]["pnl"] += t["pnl"]
        strategy_pnl[strat]["trades"] += 1
        if t["pnl"] > 0:
            strategy_pnl[strat]["wins"] += 1

    return {
        "date": today,
        "equity": round(equity, 0),
        "capital": portfolio["capital"],
        "cash": round(portfolio["cash"], 0),
        "return_pct": round((equity / portfolio["capital"] - 1) * 100, 2),
        "positions_open": len(portfolio["positions"]),
        "positions": {
            sym: {
                "direction": p["direction"],
                "qty": p["quantity"],
                "entry": p["entry_price"],
                "current": p.get("current_price", p["entry_price"]),
                "pnl": p.get("unrealized_pnl", 0),
                "pnl_pct": p.get("unrealized_pnl_pct", 0),
                "days": p.get("days_held", 0),
            }
            for sym, p in portfolio["positions"].items()
        },
        "today_pnl": round(today_pnl, 0),
        "today_trades": len(today_trades),
        "total_realized_pnl": round(total_pnl, 0),
        "total_unrealized_pnl": round(unrealized, 0),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "max_drawdown": round(portfolio.get("max_drawdown", 0), 1),
        "strategy_breakdown": strategy_pnl,
    }


def format_daily_telegram(summary: dict) -> str:
    """Format the daily summary as a rich Telegram message."""
    s = summary
    total_pnl = s["total_realized_pnl"] + s["total_unrealized_pnl"]
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"

    msg = (
        f"{pnl_emoji} <b>DAILY RECAP — {s['date']}</b>\n\n"
        f"💰 Equity: ₹{s['equity']:,.0f} ({s['return_pct']:+.1f}%)\n"
        f"💵 Cash: ₹{s['cash']:,.0f} | Deployed: ₹{s['equity'] - s['cash']:,.0f}\n"
    )

    if s["today_trades"] > 0:
        msg += f"📋 Today: {s['today_trades']} trades | P&L: ₹{s['today_pnl']:+,.0f}\n"

    msg += (
        f"\n<b>Overall:</b>\n"
        f"  Realized: ₹{s['total_realized_pnl']:+,.0f}\n"
        f"  Unrealized: ₹{s['total_unrealized_pnl']:+,.0f}\n"
        f"  Trades: {s['total_trades']} | Win Rate: {s['win_rate']:.0f}%\n"
        f"  Max Drawdown: {s['max_drawdown']:.1f}%\n"
    )

    # Open positions
    if s["positions"]:
        msg += f"\n<b>Open Positions ({s['positions_open']}):</b>\n"
        for sym, p in s["positions"].items():
            emoji = "🟢" if p["pnl"] >= 0 else "🔴"
            msg += (
                f"  {emoji} {p['direction']} {sym}: ₹{p['current']:.0f} "
                f"({p['pnl_pct']:+.1f}%) | {p['days']}d\n"
            )

    # Strategy breakdown
    if s["strategy_breakdown"]:
        msg += "\n<b>Strategy P&L:</b>\n"
        for strat, data in sorted(s["strategy_breakdown"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
            emoji = "🟢" if data["pnl"] >= 0 else "🔴"
            msg += f"  {emoji} {strat}: ₹{data['pnl']:+,.0f} ({data['trades']} trades, {wr:.0f}% win)\n"

    return msg


def reset_portfolio():
    """Reset paper portfolio to starting state."""
    default = {
        "capital": 100000,
        "cash": 100000,
        "positions": {},
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_realized_pnl": 0,
        "peak_equity": 100000,
        "max_drawdown": 0,
        "created_at": datetime.now().isoformat(),
    }
    _save_portfolio(default)
    _save_history([])
    log.info("Paper portfolio reset to ₹1,00,000")
