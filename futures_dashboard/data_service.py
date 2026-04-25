"""
Binance USDT-M futures data retrieval and simple stress simulation.
This is a local copy so the Render deployment can run without depending on
another repository checkout.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from binance.client import Client


def get_client(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Client:
    key = (api_key or "").strip() or os.environ.get("BINANCE_API_KEY", "").strip()
    secret = (api_secret or "").strip() or os.environ.get("BINANCE_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "API Key와 Secret이 필요합니다. BINANCE_API_KEY / BINANCE_API_SECRET 환경변수를 설정하세요."
        )
    return Client(key, secret)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass
class AccountSummary:
    total_wallet: float
    total_unrealized: float
    total_margin_balance: float
    available_balance: float
    total_initial_margin: float
    total_maint_margin: float
    max_withdraw: float


def fetch_account_summary(client: Client) -> Tuple[Dict[str, Any], AccountSummary]:
    raw = client.futures_account()
    s = AccountSummary(
        total_wallet=_f(raw.get("totalWalletBalance")),
        total_unrealized=_f(raw.get("totalUnrealizedProfit")),
        total_margin_balance=_f(raw.get("totalMarginBalance")),
        available_balance=_f(raw.get("availableBalance")),
        total_initial_margin=_f(raw.get("totalInitialMargin")),
        total_maint_margin=_f(raw.get("totalMaintMargin")),
        max_withdraw=_f(raw.get("maxWithdrawAmount")),
    )
    return raw, s


def fetch_positions(client: Client) -> List[Dict[str, Any]]:
    rows = client.futures_position_information()
    out = []
    for p in rows:
        if _f(p.get("positionAmt")) == 0:
            continue
        out.append(p)
    out.sort(key=lambda x: abs(_f(x.get("notional"))), reverse=True)
    return out


def fetch_open_orders(client: Client) -> List[Dict[str, Any]]:
    return client.futures_get_open_orders() or []


def get_usdm_trading_symbols(client: Client) -> List[str]:
    info = client.futures_exchange_info()
    symbols: List[str] = []
    for sym in info.get("symbols", []) or []:
        if sym.get("status") != "TRADING":
            continue
        if sym.get("quoteAsset") != "USDT":
            continue
        if sym.get("contractType") != "PERPETUAL":
            continue
        symbol = sym.get("symbol")
        if symbol:
            symbols.append(symbol)
    return sorted(set(symbols))


def get_usdm_symbol_rules(client: Client) -> Dict[str, Dict[str, float]]:
    info = client.futures_exchange_info()
    rules: Dict[str, Dict[str, float]] = {}
    for sym in info.get("symbols", []) or []:
        if sym.get("status") != "TRADING":
            continue
        if sym.get("quoteAsset") != "USDT":
            continue
        if sym.get("contractType") != "PERPETUAL":
            continue
        symbol = sym.get("symbol")
        if not symbol:
            continue
        filters = {f.get("filterType"): f for f in sym.get("filters", []) or []}
        lot = filters.get("LOT_SIZE", {}) or {}
        price = filters.get("PRICE_FILTER", {}) or {}
        rules[symbol] = {
            "step_size": _f(lot.get("stepSize"), 0.0),
            "min_qty": _f(lot.get("minQty"), 0.0),
            "tick_size": _f(price.get("tickSize"), 0.0),
        }
    return rules


def normalize_order_quantity(client: Client, symbol: str, quantity: float) -> float:
    quantity = abs(_f(quantity))
    if quantity <= 0:
        return 0.0
    rules = get_usdm_symbol_rules(client).get(symbol, {})
    step_size = _f(rules.get("step_size"), 0.0)
    min_qty = _f(rules.get("min_qty"), 0.0)
    if step_size <= 0:
        rounded = quantity
    else:
        rounded = float((Decimal(str(quantity)) / Decimal(str(step_size))).to_integral_value(rounding=ROUND_DOWN) * Decimal(str(step_size)))
    if rounded < min_qty:
        return 0.0
    return float(rounded)


def cancel_open_orders_for_symbols(client: Client, symbols: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for sym in sorted({s for s in symbols if s}):
        try:
            results.append({"symbol": sym, "result": client.futures_cancel_all_open_orders(symbol=sym)})
        except Exception as exc:
            results.append({"symbol": sym, "error": str(exc)})
    return results


def place_market_order(client: Client, symbol: str, side: str, quantity: float, *, reduce_only: bool = False) -> Dict[str, Any]:
    qty = normalize_order_quantity(client, symbol, quantity)
    if qty <= 0:
        raise ValueError(f"{symbol} 주문 수량이 최소 단위보다 작습니다.")
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side.upper(),
        "quantity": f"{qty:.16f}".rstrip("0").rstrip("."),
        "reduceOnly": "true" if reduce_only else "false",
    }
    return client.futures_create_order(**payload)


def price_map_for_symbols(client: Client, symbols: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym in sorted(set(symbols)):
        try:
            t = client.futures_symbol_ticker(symbol=sym)
            prices[sym] = _f(t.get("price"))
        except Exception:
            prices[sym] = 0.0
    return prices


def get_collateral_assets(client: Client, account_raw: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
    if account_raw is None:
        account_raw, _ = fetch_account_summary(client)
    assets = account_raw.get("assets", []) or []
    collateral: Dict[str, Dict[str, float]] = {}
    for asset in assets:
        name = asset.get("asset", "")
        wallet_balance = _f(asset.get("walletBalance"))
        unrealized = _f(asset.get("unrealizedProfit"))
        if wallet_balance <= 0 and unrealized == 0:
            continue
        if name == "USDT":
            price = 1.0
        else:
            try:
                t = client.futures_symbol_ticker(symbol=f"{name}USDT")
                price = _f(t.get("price"))
            except Exception:
                price = 0.0
        collateral[name] = {
            "wallet_balance": wallet_balance,
            "unrealized_pnl": unrealized,
            "current_price": price,
            "total_value": wallet_balance * price,
        }
    return collateral


def fetch_trades_window(client: Client, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    st = int(start.timestamp() * 1000)
    et = int(end.timestamp() * 1000)
    try:
        trades = client.futures_account_trades(startTime=st, endTime=et)
    except Exception:
        trades = []
    return sorted(trades or [], key=lambda x: x.get("time", 0))


def weekly_realized_by_symbol(client: Client, positions: List[Dict[str, Any]]) -> Dict[str, float]:
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    st = int(start.timestamp() * 1000)
    et = int(end.timestamp() * 1000)
    symbols = {p["symbol"] for p in positions}
    out: Dict[str, float] = {s: 0.0 for s in symbols}
    for sym in symbols:
        try:
            for tr in client.futures_account_trades(symbol=sym, startTime=st, endTime=et) or []:
                out[sym] += _f(tr.get("realizedPnl"))
        except Exception:
            pass
    return out


def _apply_fill_to_position(qty: float, ent: float, side: str, oqty: float, exec_px: float) -> Tuple[float, float]:
    if side == "BUY":
        if qty >= 0:
            nq = qty + oqty
            if nq <= 1e-12:
                return 0.0, 0.0
            ne = (qty * ent + oqty * exec_px) / nq if qty > 1e-12 else exec_px
            return nq, ne
        absq = abs(qty)
        if oqty <= absq + 1e-12:
            return qty + oqty, ent
        rem = oqty - absq
        return rem, exec_px
    else:
        if qty <= 0:
            if qty >= -1e-12:
                return -oqty, exec_px
            abs_old = abs(qty)
            nq = qty - oqty
            ne = (abs_old * ent + oqty * exec_px) / (abs_old + oqty)
            return nq, ne
        if oqty <= qty + 1e-12:
            return qty - oqty, ent
        rem = oqty - qty
        return -rem, exec_px


def _apply_limit_fills(
    position_amt: float,
    entry_price: float,
    mark_price: float,
    shock_pct: float,
    symbol_orders: List[Dict[str, Any]],
) -> Tuple[float, float, List[Dict[str, Any]], float]:
    new_mark = mark_price * (1.0 + shock_pct / 100.0)
    executed: List[Dict[str, Any]] = []
    qty = position_amt
    ent = entry_price if abs(position_amt) > 1e-12 else 0.0
    exec_notional = 0.0

    def sort_key(o: Dict[str, Any]) -> float:
        return _f(o.get("price")) or _f(o.get("stopPrice")) or new_mark

    orders = sorted(symbol_orders, key=sort_key)

    for order in orders:
        otype = (order.get("type") or "").upper()
        side = (order.get("side") or "").upper()
        oqty = _f(order.get("origQty"))
        limit_px = _f(order.get("price"))
        if oqty <= 0:
            continue
        fills = False
        exec_px = new_mark
        if otype == "MARKET":
            fills = True
        elif side == "BUY" and limit_px > 0 and new_mark <= limit_px:
            fills = True
            exec_px = new_mark
        elif side == "SELL" and limit_px > 0 and new_mark >= limit_px:
            fills = True
            exec_px = new_mark

        if not fills:
            continue

        exec_notional += abs(oqty * exec_px)
        executed.append({"side": side, "qty": oqty, "price": exec_px, "value": abs(oqty * exec_px)})
        qty, ent = _apply_fill_to_position(qty, ent, side, oqty, exec_px)

    if abs(qty) < 1e-12:
        qty, ent = 0.0, 0.0

    return qty, ent, executed, exec_notional


def stress_scenario(
    shock_pct: float,
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    mark_prices: Dict[str, float],
    collateral: Dict[str, Dict[str, float]],
    simulate_fills: bool,
) -> Dict[str, Any]:
    orders_by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for o in open_orders:
        orders_by_sym.setdefault(o["symbol"], []).append(o)

    pos_detail: Dict[str, Any] = {}
    total_pos_value = 0.0
    total_upl = 0.0
    total_exec_n = 0.0
    total_exec_count = 0

    for p in positions:
        sym = p["symbol"]
        amt0 = _f(p.get("positionAmt"))
        entry0 = _f(p.get("entryPrice"))
        mark0 = mark_prices.get(sym) or _f(p.get("markPrice"))
        if mark0 <= 0:
            continue

        if simulate_fills:
            amt, ent, ex_list, ex_val = _apply_limit_fills(amt0, entry0, mark0, shock_pct, orders_by_sym.get(sym, []))
            total_exec_count += len(ex_list)
            total_exec_n += ex_val
        else:
            amt, ent, ex_list, ex_val = amt0, entry0, [], 0.0

        new_mark = mark0 * (1.0 + shock_pct / 100.0)
        if abs(amt) < 1e-12:
            upl = 0.0
            pv = 0.0
        else:
            if amt > 0:
                upl = amt * (new_mark - ent)
            else:
                upl = abs(amt) * (ent - new_mark)
            pv = abs(amt) * new_mark

        total_pos_value += pv
        total_upl += upl
        pos_detail[sym] = {
            "position_amt": amt,
            "entry_price": ent,
            "shocked_mark": new_mark,
            "position_value": pv,
            "unrealized_pnl": upl,
            "executed_orders": ex_list,
        }

    coll_value = 0.0
    for name, info in collateral.items():
        bal = info["wallet_balance"]
        px = info["current_price"]
        if name == "USDT":
            coll_value += bal * 1.0
        else:
            coll_value += bal * px * (1.0 + shock_pct / 100.0)

    wallet_usdt_equiv = coll_value
    equity_proxy = wallet_usdt_equiv + total_upl
    loss_ratio = (abs(total_upl) / wallet_usdt_equiv * 100.0) if wallet_usdt_equiv > 0 else 0.0

    risk = "안전"
    if loss_ratio >= 100:
        risk = "청산 위험(단순모델)"
    elif loss_ratio >= 80:
        risk = "매우 위험"
    elif loss_ratio >= 60:
        risk = "위험"
    elif loss_ratio >= 40:
        risk = "주의"

    return {
        "shock_pct": shock_pct,
        "total_position_value": total_pos_value,
        "total_unrealized_pnl": total_upl,
        "collateral_usdt_equiv": wallet_usdt_equiv,
        "equity_proxy": equity_proxy,
        "loss_to_collateral_pct": loss_ratio,
        "risk_label": risk,
        "executed_order_count": total_exec_count,
        "executed_notional": total_exec_n,
        "positions": pos_detail,
    }


def build_stress_curve(
    shocks: List[float],
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    mark_prices: Dict[str, float],
    collateral: Dict[str, Dict[str, float]],
    simulate_fills: bool,
) -> pd.DataFrame:
    rows = []
    for s in shocks:
        r = stress_scenario(s, positions, open_orders, mark_prices, collateral, simulate_fills)
        rows.append(
            {
                "shock_pct": s,
                "equity_proxy": r["equity_proxy"],
                "total_unrealized_pnl": r["total_unrealized_pnl"],
                "collateral_usdt_equiv": r["collateral_usdt_equiv"],
                "loss_to_collateral_pct": r["loss_to_collateral_pct"],
                "total_position_value": r["total_position_value"],
            }
        )
    return pd.DataFrame(rows)


def positions_to_dataframe(positions: List[Dict[str, Any]], realized_map: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(positions, 1):
        qty = _f(p.get("positionAmt"))
        entry = _f(p.get("entryPrice"))
        mark = _f(p.get("markPrice"))
        notional = qty * mark
        upl = _f(p.get("unrealizedPnl"))
        if upl == 0 and qty != 0:
            upl = qty * (mark - entry) if qty > 0 else abs(qty) * (entry - mark)
        rows.append(
            {
                "No": i,
                "symbol": p.get("symbol"),
                "position": qty,
                "entry": entry,
                "mark": mark,
                "notional": notional,
                "unrealized_pnl": upl,
                "realized_7d": realized_map.get(p.get("symbol"), 0.0),
                "liq_price": _f(p.get("liquidationPrice")),
                "leverage": _f(p.get("leverage")),
                "margin_type": p.get("marginType", ""),
            }
        )
    return pd.DataFrame(rows)


def open_orders_to_dataframe(client: Client, orders: List[Dict[str, Any]], tz_name: str = "Asia/Seoul") -> pd.DataFrame:
    tz = pytz.timezone(tz_name)
    syms = [o["symbol"] for o in orders]
    px = price_map_for_symbols(client, syms)
    rows = []
    for o in orders:
        sym = o["symbol"]
        cp = px.get(sym, 0.0)
        q = _f(o.get("origQty"))
        limit_p = _f(o.get("price"))
        val = q * cp if cp > 0 else q * limit_p
        ot = datetime.utcfromtimestamp(o.get("time", 0) / 1000.0).replace(tzinfo=pytz.utc)
        rows.append(
            {
                "symbol": sym,
                "side": o.get("side"),
                "type": o.get("type"),
                "qty": q,
                "price": limit_p or None,
                "stop": _f(o.get("stopPrice")) or None,
                "mark": cp,
                "est_value_usdt": val,
                "time_local": ot.astimezone(tz).strftime("%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(rows)


def trades_to_dataframe(trades: List[Dict[str, Any]], price_map: Dict[str, float], tz_name: str = "Asia/Seoul") -> pd.DataFrame:
    tz = pytz.timezone(tz_name)
    rows = []
    for t in trades:
        sym = t["symbol"]
        q = _f(t.get("qty"))
        price = _f(t.get("price"))
        cp = price_map.get(sym, 0.0)
        ut = datetime.utcfromtimestamp(t.get("time", 0) / 1000.0).replace(tzinfo=pytz.utc)
        rows.append(
            {
                "symbol": sym,
                "side": t.get("side"),
                "qty": q,
                "price": price,
                "mark": cp,
                "value_usdt": q * price,
                "realized_pnl": _f(t.get("realizedPnl")),
                "time_ms": int(t.get("time", 0)),
                "time_local": ut.astimezone(tz).strftime("%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(rows)


def concentration_hhi(notional_by_symbol: Dict[str, float]) -> float:
    total = sum(notional_by_symbol.values())
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in notional_by_symbol.values())


def order_distance_stats(orders: List[Dict[str, Any]], marks: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for o in orders:
        sym = o["symbol"]
        m = marks.get(sym, 0.0)
        lp = _f(o.get("price"))
        if m <= 0 or lp <= 0:
            continue
        side = (o.get("side") or "").upper()
        if side == "BUY":
            dist_pct = (m - lp) / m * 100.0
        else:
            dist_pct = (lp - m) / m * 100.0
        rows.append({"symbol": sym, "side": side, "limit": lp, "mark": m, "dist_pct": dist_pct})
    return pd.DataFrame(rows)


def load_full_snapshot(client: Client) -> Dict[str, Any]:
    account_raw, summary = fetch_account_summary(client)
    positions = fetch_positions(client)
    open_orders = fetch_open_orders(client)
    collateral = get_collateral_assets(client, account_raw)
    syms = [p["symbol"] for p in positions] + [o["symbol"] for o in open_orders]
    marks = price_map_for_symbols(client, syms)
    for p in positions:
        sym = p["symbol"]
        if sym not in marks or marks[sym] <= 0:
            marks[sym] = _f(p.get("markPrice"))

    realized_7d = weekly_realized_by_symbol(client, positions)
    now = datetime.utcnow()
    trades_recent = fetch_trades_window(client, now - timedelta(days=3), now)

    pos_df = positions_to_dataframe(positions, realized_7d)
    oo_df = open_orders_to_dataframe(client, open_orders)
    tr_syms = list({t["symbol"] for t in trades_recent})
    tr_px = price_map_for_symbols(client, tr_syms)
    tr_df = trades_to_dataframe(trades_recent, tr_px)

    notional_map = {r["symbol"]: abs(float(r["notional"])) for _, r in pos_df.iterrows()} if len(pos_df) else {}
    hhi = concentration_hhi(notional_map)

    maint_buffer = summary.total_margin_balance - summary.total_maint_margin
    margin_ratio_pct = (
        (summary.total_maint_margin / summary.total_margin_balance * 100.0)
        if summary.total_margin_balance > 0
        else 0.0
    )

    return {
        "account_raw": account_raw,
        "summary": summary,
        "positions": positions,
        "open_orders": open_orders,
        "collateral": collateral,
        "mark_prices": marks,
        "realized_7d": realized_7d,
        "pos_df": pos_df,
        "oo_df": oo_df,
        "trades_recent": trades_recent,
        "tr_df": tr_df,
        "hhi": hhi,
        "maint_buffer": maint_buffer,
        "margin_ratio_pct": margin_ratio_pct,
    }
