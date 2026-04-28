"""
factors/rebalancer.py

Monthly rebalance job for the Factor Engine.

This is the LIVE counterpart of factors/backtest.py.
Instead of simulating, it:
  1. Ranks the current universe
  2. Compares to current portfolio
  3. Generates buy/sell actions
  4. Executes via portfolio_tracker (paper) or Kite (live)
  5. Sends Telegram summary

Schedule: Run on the LAST TRADING DAY of each month at 2:30 PM IST.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from factors.base import RebalanceAction, PortfolioHolding
from factors.composite import CompositeFactorEngine, momentum_quality_preset
from factors.universe import get_universe, get_sector_map, SECTOR_MAP


# ─── State File ───────────────────────────────────────────────

FACTOR_PORTFOLIO_PATH = Path("data/factor_portfolio.json")


def _load_factor_portfolio() -> dict:
    """Load current factor portfolio state."""
    if FACTOR_PORTFOLIO_PATH.exists():
        with open(FACTOR_PORTFOLIO_PATH) as f:
            return json.load(f)
    return {
        "holdings": {},        # {symbol: {shares, entry_price, entry_date, weight}}
        "cash": 100_000,
        "initial_capital": 100_000,
        "last_rebalance": None,
        "rebalance_count": 0,
        "history": [],         # Past rebalance summaries
    }


def _save_factor_portfolio(portfolio: dict):
    """Save factor portfolio state."""
    FACTOR_PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTOR_PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


# ─── Rebalance Logic ──────────────────────────────────────────

def run_monthly_rebalance(
    use_kite: bool = True,
    dry_run: bool = True,
    capital: float | None = None,
) -> dict:
    """
    Execute monthly factor rebalance.

    Args:
        use_kite: If True, fetch live prices from Kite. If False, use yfinance.
        dry_run: If True, only show what WOULD happen. If False, execute.
        capital: Override capital (otherwise uses portfolio state).

    Returns:
        Summary dict with actions, new holdings, metrics.
    """
    print(f"\n{'='*60}")
    print(f"  FACTOR ENGINE — MONTHLY REBALANCE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    portfolio = _load_factor_portfolio()
    current_capital = capital or portfolio.get("initial_capital", 100_000)

    # Get current prices
    symbols = get_universe("nifty200")
    price_data = _fetch_current_prices(symbols, use_kite)
    fund_data = _load_fundamentals()

    if len(price_data) < 50:
        print(f"  ERROR: Only {len(price_data)} stocks loaded. Need 50+. Aborting.")
        return {"error": "insufficient_data"}

    # Compute current portfolio value
    current_holdings = portfolio.get("holdings", {})
    port_value = float(portfolio.get("cash", current_capital))
    for sym, info in current_holdings.items():
        if sym in price_data and not price_data[sym].empty:
            price = float(price_data[sym]["close"].iloc[-1])
            port_value += info.get("shares", 0) * price

    print(f"  Portfolio value: ₹{port_value:,.0f}")
    print(f"  Current holdings: {len(current_holdings)} stocks")

    # Build engine and rank
    engine = CompositeFactorEngine(
        factors=momentum_quality_preset(),
        top_n=20,
        weighting="equal",
    )

    selected = engine.select_portfolio(
        price_data=price_data,
        fundamental_data=fund_data,
        sector_map=get_sector_map(),
    )
    target_weights = engine.compute_weights(selected)

    print(f"  Top 20 selected. Computing trades...\n")

    # Compute actions
    actions = _compute_rebalance_actions(
        current_holdings=current_holdings,
        target_weights=target_weights,
        price_data=price_data,
        portfolio_value=port_value,
        cost_pct=0.002,
    )

    # Print actions
    buys = [a for a in actions if a.action in ("BUY", "INCREASE")]
    sells = [a for a in actions if a.action in ("SELL", "DECREASE")]
    holds = [a for a in actions if a.action == "HOLD"]

    if sells:
        print(f"  SELL ({len(sells)}):")
        for a in sells:
            print(f"    {a.symbol:15s} | {a.shares_delta:+5d} shares | cost ₹{a.estimated_cost:,.0f}")

    if buys:
        print(f"\n  BUY ({len(buys)}):")
        for a in buys:
            print(f"    {a.symbol:15s} | {a.shares_delta:+5d} shares | cost ₹{a.estimated_cost:,.0f}")

    if holds:
        print(f"\n  HOLD ({len(holds)}): {', '.join(a.symbol for a in holds)}")

    total_cost = sum(a.estimated_cost for a in actions)
    turnover = sum(abs(a.shares_delta) * _get_price(a.symbol, price_data) for a in actions if a.action != "HOLD")
    print(f"\n  Turnover: ₹{turnover:,.0f} ({turnover/port_value*100:.1f}%)")
    print(f"  Transaction costs: ₹{total_cost:,.0f}")

    # Show top 20 rankings
    print(f"\n  {'Rank':<5} {'Symbol':<15} {'Composite Z':<13} {'Sector':<12}")
    print(f"  {'─'*50}")
    for cs in selected:
        sector = SECTOR_MAP.get(cs.symbol, "Unknown")
        print(f"  {cs.rank:<5} {cs.symbol:<15} {cs.composite_z:+.3f}        {sector}")

    if not dry_run:
        _execute_rebalance(portfolio, actions, price_data, port_value)
        print(f"\n  ✅ Rebalance executed. Portfolio saved.")

    # Build summary
    summary = {
        "date": datetime.now().isoformat(),
        "portfolio_value": round(port_value, 2),
        "holdings_before": len(current_holdings),
        "holdings_after": len(target_weights),
        "buys": len(buys),
        "sells": len(sells),
        "holds": len(holds),
        "turnover_pct": round(turnover / port_value * 100, 1) if port_value > 0 else 0,
        "transaction_costs": round(total_cost, 2),
        "top_20": [{"symbol": cs.symbol, "rank": cs.rank, "z": cs.composite_z} for cs in selected],
        "dry_run": dry_run,
    }

    return summary


def _fetch_current_prices(
    symbols: list[str],
    use_kite: bool,
) -> dict[str, object]:
    """Fetch prices either from Kite (live) or yfinance (backtest)."""
    if use_kite:
        try:
            from core.auth import get_kite
            from core.data import fetch_candles
            from utils import load_json
            from config.settings import cfg

            kite = get_kite()
            cache = load_json(cfg.DATA_DIR / "instrument_cache.json", default={})
            tokens = {i["symbol"]: i["token"] for i in cache.get("nse", [])}

            import pandas as pd
            price_data = {}
            for sym in symbols:
                token = tokens.get(sym, 0)
                if token:
                    try:
                        df = fetch_candles(token, interval="day", days=300)
                        if not df.empty:
                            price_data[sym] = df
                    except Exception:
                        pass

            if len(price_data) > 50:
                return price_data
            print(f"  Kite returned only {len(price_data)} stocks, falling back to yfinance")
        except Exception as e:
            print(f"  Kite failed ({e}), falling back to yfinance")

    # Fallback: yfinance
    from factors.universe import fetch_universe_prices
    return fetch_universe_prices(symbols, years=2)


def _load_fundamentals() -> dict[str, dict]:
    """Load cached fundamental data."""
    from factors.universe import load_fundamental_data
    return load_fundamental_data()


def _get_price(symbol: str, price_data: dict) -> float:
    """Get latest close price for a symbol."""
    if symbol in price_data and not price_data[symbol].empty:
        return float(price_data[symbol]["close"].iloc[-1])
    return 0.0


def _compute_rebalance_actions(
    current_holdings: dict,
    target_weights: dict[str, float],
    price_data: dict,
    portfolio_value: float,
    cost_pct: float,
) -> list[RebalanceAction]:
    """Compute the buy/sell/hold actions for rebalancing."""
    all_symbols = set(list(current_holdings.keys()) + list(target_weights.keys()))
    actions = []

    for sym in sorted(all_symbols):
        current_shares = current_holdings.get(sym, {}).get("shares", 0) if isinstance(current_holdings.get(sym), dict) else 0
        target_weight = target_weights.get(sym, 0)
        price = _get_price(sym, price_data)

        if price <= 0:
            continue

        current_value = current_shares * price
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
        target_value = portfolio_value * target_weight
        target_shares = int(target_value / price) if price > 0 else 0
        delta = target_shares - current_shares

        if delta == 0:
            action_type = "HOLD" if current_shares > 0 else "SKIP"
        elif current_shares == 0 and target_shares > 0:
            action_type = "BUY"
        elif current_shares > 0 and target_shares == 0:
            action_type = "SELL"
        elif delta > 0:
            action_type = "INCREASE"
        else:
            action_type = "DECREASE"

        if action_type == "SKIP":
            continue

        trade_value = abs(delta) * price
        cost = trade_value * cost_pct

        actions.append(RebalanceAction(
            symbol=sym,
            action=action_type,
            target_weight=round(target_weight, 4),
            current_weight=round(current_weight, 4),
            shares_delta=delta,
            estimated_cost=round(cost, 2),
        ))

    return actions


def _execute_rebalance(
    portfolio: dict,
    actions: list[RebalanceAction],
    price_data: dict,
    portfolio_value: float,
):
    """Execute rebalance: update portfolio state file."""
    cash = float(portfolio.get("cash", 0))
    holdings = portfolio.get("holdings", {})
    today = datetime.now().strftime("%Y-%m-%d")

    for action in actions:
        price = _get_price(action.symbol, price_data)
        if price <= 0 or action.action == "HOLD":
            continue

        if action.action in ("SELL", "DECREASE"):
            sell_shares = abs(action.shares_delta)
            sell_value = sell_shares * price
            cash += sell_value - action.estimated_cost
            current = holdings.get(action.symbol, {}).get("shares", 0)
            new_shares = current - sell_shares
            if new_shares <= 0:
                holdings.pop(action.symbol, None)
            else:
                holdings[action.symbol]["shares"] = new_shares

        elif action.action in ("BUY", "INCREASE"):
            buy_value = action.shares_delta * price
            cash -= buy_value + action.estimated_cost
            if action.symbol not in holdings:
                holdings[action.symbol] = {
                    "shares": action.shares_delta,
                    "entry_price": price,
                    "entry_date": today,
                    "weight": action.target_weight,
                }
            else:
                # Average up
                old = holdings[action.symbol]
                old_shares = old.get("shares", 0)
                new_shares = old_shares + action.shares_delta
                old_cost = old_shares * old.get("entry_price", price)
                new_cost = action.shares_delta * price
                avg_price = (old_cost + new_cost) / new_shares if new_shares > 0 else price
                holdings[action.symbol] = {
                    "shares": new_shares,
                    "entry_price": round(avg_price, 2),
                    "entry_date": today,
                    "weight": action.target_weight,
                }

    portfolio["holdings"] = holdings
    portfolio["cash"] = round(cash, 2)
    portfolio["last_rebalance"] = today
    portfolio["rebalance_count"] = portfolio.get("rebalance_count", 0) + 1

    _save_factor_portfolio(portfolio)


def format_rebalance_telegram(summary: dict) -> str:
    """Format rebalance summary for Telegram."""
    lines = [
        f"📊 *Factor Engine Rebalance*",
        f"Portfolio: ₹{summary['portfolio_value']:,.0f}",
        f"",
        f"Buys: {summary['buys']} | Sells: {summary['sells']} | Holds: {summary['holds']}",
        f"Turnover: {summary['turnover_pct']:.1f}%",
        f"Costs: ₹{summary['transaction_costs']:,.0f}",
        f"",
        f"*Top 5 Holdings:*",
    ]
    for item in summary.get("top_20", [])[:5]:
        lines.append(f"  {item['rank']}. {item['symbol']} (z={item['z']:+.2f})")

    if summary.get("dry_run"):
        lines.append(f"\n⚠️ DRY RUN — no trades executed")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Factor Engine Rebalancer")
    parser.add_argument("--live", action="store_true", help="Execute trades (not dry run)")
    parser.add_argument("--no-kite", action="store_true", help="Use yfinance instead of Kite")
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()

    summary = run_monthly_rebalance(
        use_kite=not args.no_kite,
        dry_run=not args.live,
        capital=args.capital,
    )

    if "error" not in summary:
        msg = format_rebalance_telegram(summary)
        print(f"\n{'='*60}")
        print(f"  TELEGRAM MESSAGE:")
        print(f"{'='*60}")
        print(msg)
