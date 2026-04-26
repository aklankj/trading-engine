"""
Telegram bot for trade signal delivery and approval flow.

Provides:
- send_signal(): Send a signal with approve/reject buttons
- send_alert(): Send text alerts (regime changes, daily recap)
- Callback handler for approval/rejection buttons
"""

import requests
import json
from config.settings import cfg
from utils.logger import log
from utils import fmt_inr, fmt_pct


BASE_URL = f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}"


def send_message(text: str, parse_mode: str = "HTML", reply_markup: dict = None) -> bool:
    """Send a text message via Telegram."""
    if not cfg.TELEGRAM_BOT_TOKEN or not cfg.TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured, skipping message")
        return False

    payload = {
        "chat_id": cfg.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)

    try:
        resp = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        log.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def send_signal(order: dict) -> bool:
    """
    Send a trading signal with approve/reject inline buttons.
    Called when TRADING_MODE is 'approval'.
    """
    direction_emoji = "🟢" if order["direction"] == "BUY" else "🔴"
    regime = order.get("regime", "Unknown")

    text = (
        f"{direction_emoji} <b>TRADE SIGNAL: {order['direction']} {order['symbol']}</b>\n"
        f"\n"
        f"📊 <b>Signal Details</b>\n"
        f"  Regime: <code>{regime}</code>\n"
        f"  Composite: <code>{order.get('composite_signal', 0):+.3f}</code>\n"
        f"  Agreement: <code>{order.get('agreement', 0):.0%}</code>\n"
        f"\n"
        f"💰 <b>Position</b>\n"
        f"  Qty: <code>{order['quantity']}</code> shares\n"
        f"  Value: <code>{fmt_inr(order.get('value', 0))}</code>\n"
        f"  Capital: <code>{order.get('pct_of_capital', 0):.1%}</code>\n"
        f"\n"
        f"🎯 <b>Risk Management</b>\n"
        f"  Stop Loss: <code>₹{order.get('stop_loss', 0):.2f}</code>\n"
        f"  Target: <code>₹{order.get('target', 0):.2f}</code>\n"
        f"  R:R = <code>{order.get('risk_reward', 0):.2f}</code>\n"
    )

    order_id = order.get("order_id", "unknown")
    markup = {
        "inline_keyboard": [[
            {"text": "✅ APPROVE", "callback_data": f"approve:{order_id}"},
            {"text": "❌ REJECT", "callback_data": f"reject:{order_id}"},
        ]]
    }

    return send_message(text, reply_markup=markup)


def send_regime_alert(symbol: str, old_regime: str, new_regime: str, state) -> bool:
    """Send a regime change alert."""
    regime_emojis = {
        "Bull": "🐂", "Bear": "🐻", "Sideways": "➡️",
        "HighVol": "🌊", "Recovery": "🌱", "Unknown": "❓",
    }

    text = (
        f"🔄 <b>REGIME CHANGE: {symbol}</b>\n"
        f"\n"
        f"  {regime_emojis.get(old_regime, '')} {old_regime} → "
        f"{regime_emojis.get(new_regime, '')} <b>{new_regime}</b>\n"
        f"\n"
        f"  Ann. Return: <code>{state.ann_return:.1%}</code>\n"
        f"  Ann. Volatility: <code>{state.ann_volatility:.1%}</code>\n"
        f"  Confidence: <code>{state.confidence:.0%}</code>\n"
        f"  Smoothed Signal: <code>{state.smoothed_signal:+.2f}</code>\n"
    )
    return send_message(text)


def send_daily_recap(recap: dict) -> bool:
    """Send end-of-day P&L and summary."""
    pnl = recap.get("daily_pnl", 0)
    pnl_emoji = "📈" if pnl >= 0 else "📉"

    text = (
        f"{pnl_emoji} <b>DAILY RECAP — {recap.get('date', 'Today')}</b>\n"
        f"\n"
        f"💰 P&L: <code>{fmt_inr(pnl)}</code> ({fmt_pct(recap.get('daily_pnl_pct', 0))})\n"
        f"📊 Regime: <code>{recap.get('regime', 'Unknown')}</code>\n"
        f"📋 Signals: <code>{recap.get('total_signals', 0)}</code> "
        f"(executed: {recap.get('executed', 0)})\n"
        f"\n"
        f"🏆 <b>Strategy Performance</b>\n"
    )

    for strat, perf in recap.get("strategy_performance", {}).items():
        text += f"  {strat}: <code>{perf:+.3f}</code>\n"

    text += (
        f"\n"
        f"📊 <b>Portfolio</b>\n"
        f"  Capital: <code>{fmt_inr(recap.get('total_capital', 0))}</code>\n"
        f"  Deployed: <code>{fmt_inr(recap.get('deployed', 0))}</code>\n"
        f"  Cash: <code>{fmt_inr(recap.get('cash', 0))}</code>\n"
    )

    return send_message(text)


def send_fundamental_alert(stock: dict) -> bool:
    """Send a fundamental buy opportunity alert."""
    text = (
        f"🏛️ <b>FUNDAMENTAL OPPORTUNITY: {stock['name']}</b>\n"
        f"\n"
        f"  Sector: <code>{stock.get('sector', 'N/A')}</code>\n"
        f"  Quality Score: <code>{stock.get('score', 0)}/100</code>\n"
        f"  ROCE: <code>{stock.get('roce', 0):.1f}%</code>\n"
        f"  Drop from 52W High: <code>{stock.get('drop_pct', 0):.1f}%</code>\n"
        f"\n"
        f"  Current Price: <code>₹{stock.get('price', 0):,.2f}</code>\n"
        f"  52W High: <code>₹{stock.get('high_52w', 0):,.2f}</code>\n"
        f"\n"
        f"  📝 {stock.get('note', '')}\n"
    )
    return send_message(text)


def send_research_alert(paper: dict) -> bool:
    """Send a new research paper/strategy discovery alert."""
    text = (
        f"📚 <b>NEW STRATEGY FOUND</b>\n"
        f"\n"
        f"  Title: <i>{paper.get('title', 'N/A')}</i>\n"
        f"  Source: <code>{paper.get('source', 'N/A')}</code>\n"
        f"  Reported Sharpe: <code>{paper.get('sharpe', 'N/A')}</code>\n"
        f"\n"
        f"  💡 {paper.get('insight', '')}\n"
        f"\n"
        f"  🔗 {paper.get('url', 'N/A')}\n"
    )
    return send_message(text)
