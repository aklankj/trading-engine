"""
Order execution — places orders via Kite or logs them for paper trading.

Respects the TRADING_MODE from config:
  - paper:     Log signal to CSV, no real orders
  - approval:  Send to Telegram for human approval, then execute
  - semi_auto: Auto-execute within strict limits, notify via Telegram
"""

from datetime import datetime
from kiteconnect import KiteConnect

from config.settings import cfg
from core.auth import get_kite
from core.risk_gate import RiskDecision
from core.meta_allocator import CompositeSignal
from utils.logger import log
from utils import append_csv, load_json, save_json, now_ist, fmt_inr


def execute_signal(
    symbol: str,
    exchange: str,
    signal: CompositeSignal,
    risk: RiskDecision,
    product: str = "MIS",
) -> dict:
    """
    Execute a trading signal based on current mode.

    Returns dict with order details or paper trade log.
    """
    if not risk.approved:
        log.warning(f"Signal for {symbol} rejected by risk gate: {risk.reason}")
        return {"status": "rejected", "reason": risk.reason}

    # Build order params
    order = {
        "timestamp": now_ist().isoformat(),
        "symbol": symbol,
        "exchange": exchange,
        "direction": signal.direction,
        "quantity": risk.position_size,
        "price": 0,  # Market order
        "product": product,
        "stop_loss": risk.stop_loss,
        "target": risk.target,
        "risk_reward": risk.risk_reward,
        "regime": signal.regime.regime,
        "composite_signal": round(signal.signal, 4),
        "agreement": round(signal.agreement_pct, 2),
        "value": round(risk.position_value, 2),
        "pct_of_capital": round(risk.pct_of_capital, 4),
    }

    mode = cfg.TRADING_MODE

    if mode == "paper":
        return _paper_trade(order)
    elif mode == "approval":
        return _queue_for_approval(order)
    elif mode == "semi_auto":
        return _semi_auto_execute(order)
    else:
        log.error(f"Unknown trading mode: {mode}")
        return {"status": "error", "reason": f"Unknown mode: {mode}"}


def _paper_trade(order: dict) -> dict:
    """Log the signal to CSV without placing a real order."""
    order["status"] = "paper"
    order["order_id"] = f"PAPER-{now_ist().strftime('%Y%m%d%H%M%S')}"

    append_csv(cfg.SIGNAL_LOG, order)
    _update_trade_log(order)

    log.bind(trade=True).info(
        f"📝 PAPER TRADE: {order['direction']} {order['quantity']} {order['symbol']} | "
        f"Value: {fmt_inr(order['value'])} | Regime: {order['regime']} | "
        f"Signal: {order['composite_signal']:+.3f}"
    )
    return order


def _queue_for_approval(order: dict) -> dict:
    """Save order to pending queue — Telegram bot picks it up."""
    order["status"] = "pending_approval"
    order["order_id"] = f"PENDING-{now_ist().strftime('%Y%m%d%H%M%S')}"

    # Save to pending orders file
    pending_path = cfg.DATA_DIR / "pending_orders.json"
    pending = load_json(pending_path, default={"orders": []})
    pending["orders"].append(order)
    save_json(pending_path, pending)

    log.bind(trade=True).info(
        f"⏳ QUEUED FOR APPROVAL: {order['direction']} {order['quantity']} {order['symbol']}"
    )
    return order


def _semi_auto_execute(order: dict) -> dict:
    """Auto-execute if within strict limits, otherwise queue for approval."""
    # Semi-auto only for small positions with strong consensus
    if (
        order["pct_of_capital"] <= 0.03
        and abs(order["composite_signal"]) >= 0.7
        and order["agreement"] >= 0.6
    ):
        return _place_real_order(order)
    else:
        log.info(f"Signal too large or weak for semi-auto, queuing for approval")
        return _queue_for_approval(order)


def _place_real_order(order: dict) -> dict:
    """Place a real order via Kite Connect."""
    kite = get_kite()

    transaction = (
        kite.TRANSACTION_TYPE_BUY
        if order["direction"] == "BUY"
        else kite.TRANSACTION_TYPE_SELL
    )

    product_map = {
        "MIS": kite.PRODUCT_MIS,
        "NRML": kite.PRODUCT_NRML,
        "CNC": kite.PRODUCT_CNC,
    }
    kite_product = product_map.get(order["product"], kite.PRODUCT_MIS)

    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            tradingsymbol=order["symbol"],
            exchange=order["exchange"],
            transaction_type=transaction,
            quantity=order["quantity"],
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite_product,
            validity=kite.VALIDITY_DAY,
        )

        order["status"] = "executed"
        order["order_id"] = str(order_id)

        # Place stop-loss order
        sl_transaction = (
            kite.TRANSACTION_TYPE_SELL
            if order["direction"] == "BUY"
            else kite.TRANSACTION_TYPE_BUY
        )
        try:
            sl_order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                tradingsymbol=order["symbol"],
                exchange=order["exchange"],
                transaction_type=sl_transaction,
                quantity=order["quantity"],
                order_type=kite.ORDER_TYPE_SL,
                product=kite_product,
                validity=kite.VALIDITY_DAY,
                trigger_price=order["stop_loss"],
                price=order["stop_loss"],
            )
            order["sl_order_id"] = str(sl_order_id)
        except Exception as e:
            log.warning(f"SL order failed for {order['symbol']}: {e}")
            order["sl_order_id"] = None

        append_csv(cfg.SIGNAL_LOG, order)
        _update_trade_log(order)

        log.bind(trade=True).info(
            f"🔥 ORDER PLACED: {order['direction']} {order['quantity']} {order['symbol']} | "
            f"Order ID: {order_id} | Value: {fmt_inr(order['value'])}"
        )
        return order

    except Exception as e:
        order["status"] = "failed"
        order["error"] = str(e)
        log.error(f"Order placement failed for {order['symbol']}: {e}")
        return order


def approve_pending_order(order_id: str) -> dict:
    """Called by Telegram bot when user approves a pending order."""
    pending_path = cfg.DATA_DIR / "pending_orders.json"
    pending = load_json(pending_path, default={"orders": []})

    order = None
    remaining = []
    for o in pending.get("orders", []):
        if o.get("order_id") == order_id:
            order = o
        else:
            remaining.append(o)

    if not order:
        return {"status": "error", "reason": f"Order {order_id} not found"}

    pending["orders"] = remaining
    save_json(pending_path, pending)

    return _place_real_order(order)


def reject_pending_order(order_id: str) -> dict:
    """Called by Telegram bot when user rejects a pending order."""
    pending_path = cfg.DATA_DIR / "pending_orders.json"
    pending = load_json(pending_path, default={"orders": []})

    remaining = []
    rejected = None
    for o in pending.get("orders", []):
        if o.get("order_id") == order_id:
            o["status"] = "rejected_by_user"
            rejected = o
        else:
            remaining.append(o)

    pending["orders"] = remaining
    save_json(pending_path, pending)

    if rejected:
        append_csv(cfg.SIGNAL_LOG, rejected)
        log.bind(trade=True).info(f"❌ Order {order_id} rejected by user")
    return rejected or {"status": "not_found"}


def _update_trade_log(order: dict):
    """Append trade to persistent trade log."""
    trade_log = load_json(cfg.TRADE_LOG, default={"trades": [], "daily_pnl": {}})
    trade_log["trades"].append(order)
    save_json(cfg.TRADE_LOG, trade_log)
