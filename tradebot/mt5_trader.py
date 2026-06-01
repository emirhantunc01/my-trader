from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import BacktestConfig
from .indicators import add_common_indicators
from .mt5_adapter import ensure_symbol, latest_bid_ask
from .strategies import Strategy


DEMO_MAGIC = 260601


@dataclass
class DemoTradeDecision:
    symbol: str
    action: str
    reason: str
    signal: str = "UNKNOWN"
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    retcode: int | None = None
    comment: str = ""
    position_side: str = ""
    position_volume: float = 0.0
    position_entry: float = 0.0
    position_profit: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_demo_state() -> dict:
    return {
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "symbols": {},
    }


def load_demo_state(path: Path, reset: bool = False) -> dict:
    if reset or not path.exists():
        return default_demo_state()
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state.setdefault("created_at", _now_iso())
    state.setdefault("symbols", {})
    return state


def save_demo_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now_iso()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def _symbol_state(state: dict | None, symbol: str) -> dict | None:
    if state is None:
        return None
    symbols = state.setdefault("symbols", {})
    return symbols.setdefault(symbol, {})


def _bar_id(bar: pd.Series) -> str:
    if isinstance(bar.name, pd.Timestamp):
        return bar.name.isoformat()
    return str(bar.name)


def _signal_label(buy_signal: bool, sell_signal: bool) -> str:
    if buy_signal and sell_signal:
        return "BOTH"
    if buy_signal:
        return "BUY"
    if sell_signal:
        return "SELL"
    return "NONE"


def _fmt(value, digits: int = 5) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_signal_debug(symbol: str, bar_id: str, signal_bar: pd.Series) -> None:
    close = float(signal_bar.get("Close", 0.0))
    ema_8 = float(signal_bar.get("ema_8", 0.0))
    ema_21 = float(signal_bar.get("ema_21", 0.0))
    ema_20 = float(signal_bar.get("ema_20", 0.0))
    ema_50 = float(signal_bar.get("ema_50", 0.0))
    ema_200 = float(signal_bar.get("ema_200", 0.0))
    macd_hist = float(signal_bar.get("macd_hist", 0.0))

    checks = {
        "close>ema200": close > ema_200,
        "ema8>ema21": ema_8 > ema_21,
        "ema20>ema50": ema_20 > ema_50,
        "macd_hist>0": macd_hist > 0,
        "volume_pump": bool(signal_bar.get("volume_pump", False)),
    }
    checks_text = " ".join(f"{key}={value}" for key, value in checks.items())
    print(
        f"[SIGNAL] {symbol} bar={bar_id} "
        f"score={signal_bar.get('signal_score', 'N/A')} "
        f"buy={bool(signal_bar.get('buy_signal', False))} "
        f"sell={bool(signal_bar.get('sell_signal', False))} "
        f"close={_fmt(close)} ema8={_fmt(ema_8)} ema21={_fmt(ema_21)} "
        f"ema50={_fmt(ema_50)} ema200={_fmt(ema_200)} "
        f"adx={_fmt(signal_bar.get('adx'), 2)} "
        f"macd_hist={_fmt(macd_hist, 6)} "
        f"rsi={_fmt(signal_bar.get('rsi'), 2)} "
        f"cmf={_fmt(signal_bar.get('cmf'), 3)} "
        f"{checks_text}"
    )


def _remember_signal_state(
    symbol_state: dict | None,
    *,
    bar_id: str,
    buy_signal: bool,
    sell_signal: bool,
    has_position: bool,
    decision: DemoTradeDecision,
) -> DemoTradeDecision:
    if symbol_state is None:
        if decision.signal == "UNKNOWN":
            decision.signal = _signal_label(buy_signal, sell_signal)
        return decision
    if decision.signal == "UNKNOWN":
        decision.signal = _signal_label(buy_signal, sell_signal)
    symbol_state["last_seen_bar"] = bar_id
    symbol_state["last_buy_signal"] = bool(buy_signal)
    symbol_state["last_sell_signal"] = bool(sell_signal)
    symbol_state["has_position"] = bool(has_position)
    symbol_state["last_decision"] = {
        "timestamp": _now_iso(),
        "action": decision.action,
        "reason": decision.reason,
        "retcode": decision.retcode,
    }
    if decision.action == "BUY_SENT":
        symbol_state["last_buy_bar"] = bar_id
    if decision.action == "SHORT_SENT":
        symbol_state["last_short_bar"] = bar_id
    if decision.action == "CLOSE_SENT":
        symbol_state["last_close_bar"] = bar_id
        symbol_state["has_position"] = False
    return decision


def assert_demo_account(mt5) -> None:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Could not read MT5 account info: {mt5.last_error()}")

    data = info._asdict()
    server = str(data.get("server", ""))
    if "demo" not in server.lower():
        raise RuntimeError(
            f"Refusing to send orders because account server is not demo: {server!r}"
        )
    if data.get("trade_allowed") is False:
        raise RuntimeError("MT5 account says trading is not allowed.")


def normalize_volume(mt5, symbol: str, requested_volume: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"MT5 symbol not found: {symbol}")

    min_volume = float(info.volume_min)
    max_volume = float(info.volume_max)
    step = float(info.volume_step)
    requested = max(min_volume, min(float(requested_volume), max_volume))
    steps = math.floor((requested - min_volume) / step)
    volume = min_volume + steps * step
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(volume, precision)


def open_positions(mt5, symbol: str) -> list:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [position for position in positions if getattr(position, "magic", None) == DEMO_MAGIC]


def _with_position(decision: DemoTradeDecision, position) -> DemoTradeDecision:
    position_type = getattr(position, "type", None)
    decision.position_side = "SHORT" if position_type == 1 else "LONG"
    decision.position_volume = float(getattr(position, "volume", 0.0))
    decision.position_entry = float(getattr(position, "price_open", 0.0))
    decision.position_profit = float(getattr(position, "profit", 0.0))
    return decision


def _position_age_minutes(mt5, position) -> float:
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is not None and getattr(tick, "time", 0):
        return max(0.0, (float(tick.time) - float(getattr(position, "time", 0))) / 60.0)
    opened_at = datetime.fromtimestamp(int(getattr(position, "time", 0)), tz=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds() / 60.0)


def _filling_type(mt5, symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_RETURN
    filling_mode = int(getattr(info, "filling_mode", 0))
    for candidate in (
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN,
    ):
        if filling_mode & candidate == candidate:
            return candidate
    return mt5.ORDER_FILLING_FOK


def _send_with_filling_fallbacks(mt5, request: dict):
    tried = []
    for filling in (
        request.get("type_filling"),
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN,
    ):
        if filling in tried:
            continue
        tried.append(filling)
        request["type_filling"] = filling
        result = mt5.order_send(request)
        if result is None:
            continue
        if result.retcode != 10030:
            return result
    return result if "result" in locals() else None


def send_market_buy(
    mt5,
    symbol: str,
    volume: float,
    sl: float,
    tp: float,
    deviation: int,
) -> DemoTradeDecision:
    ensure_symbol(mt5, symbol)
    bid, ask = latest_bid_ask(mt5, symbol)
    info = mt5.symbol_info(symbol)
    digits = int(getattr(info, "digits", 5)) if info else 5
    volume = normalize_volume(mt5, symbol, volume)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": round(sl, digits) if sl > 0 else 0.0,
        "tp": round(tp, digits) if tp > 0 else 0.0,
        "deviation": deviation,
        "magic": DEMO_MAGIC,
        "comment": "my-trader demo buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, symbol),
    }
    result = _send_with_filling_fallbacks(mt5, request)
    if result is None:
        return DemoTradeDecision(
            symbol=symbol,
            action="BUY_FAILED",
            reason=str(mt5.last_error()),
            volume=volume,
            price=ask,
            sl=request["sl"],
            tp=request["tp"],
        )
    return DemoTradeDecision(
        symbol=symbol,
        action="BUY_SENT" if result.retcode == mt5.TRADE_RETCODE_DONE else "BUY_FAILED",
        reason=getattr(result, "comment", ""),
        volume=volume,
        price=ask,
        sl=request["sl"],
        tp=request["tp"],
        retcode=int(result.retcode),
        comment=str(result),
    )


def send_market_sell_short(
    mt5,
    symbol: str,
    volume: float,
    sl: float,
    tp: float,
    deviation: int,
) -> DemoTradeDecision:
    ensure_symbol(mt5, symbol)
    bid, ask = latest_bid_ask(mt5, symbol)
    info = mt5.symbol_info(symbol)
    digits = int(getattr(info, "digits", 5)) if info else 5
    volume = normalize_volume(mt5, symbol, volume)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": bid,
        "sl": round(sl, digits) if sl > 0 else 0.0,
        "tp": round(tp, digits) if tp > 0 else 0.0,
        "deviation": deviation,
        "magic": DEMO_MAGIC,
        "comment": "my-trader demo short",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, symbol),
    }
    result = _send_with_filling_fallbacks(mt5, request)
    if result is None:
        return DemoTradeDecision(
            symbol=symbol,
            action="SHORT_FAILED",
            reason=str(mt5.last_error()),
            signal="SELL",
            volume=volume,
            price=bid,
            sl=request["sl"],
            tp=request["tp"],
        )
    return DemoTradeDecision(
        symbol=symbol,
        action="SHORT_SENT" if result.retcode == mt5.TRADE_RETCODE_DONE else "SHORT_FAILED",
        reason=getattr(result, "comment", ""),
        signal="SELL",
        volume=volume,
        price=bid,
        sl=request["sl"],
        tp=request["tp"],
        retcode=int(result.retcode),
        comment=str(result),
    )


def close_position(mt5, position, deviation: int) -> DemoTradeDecision:
    symbol = position.symbol
    bid, ask = latest_bid_ask(mt5, symbol)
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = bid if order_type == mt5.ORDER_TYPE_SELL else ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(position.volume),
        "type": order_type,
        "position": int(position.ticket),
        "price": price,
        "deviation": deviation,
        "magic": DEMO_MAGIC,
        "comment": "my-trader demo close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, symbol),
    }
    result = _send_with_filling_fallbacks(mt5, request)
    if result is None:
        return DemoTradeDecision(
            symbol=symbol,
            action="CLOSE_FAILED",
            reason=str(mt5.last_error()),
            volume=float(position.volume),
            price=price,
        )
    return DemoTradeDecision(
        symbol=symbol,
        action="CLOSE_SENT" if result.retcode == mt5.TRADE_RETCODE_DONE else "CLOSE_FAILED",
        reason=getattr(result, "comment", ""),
        volume=float(position.volume),
        price=price,
        retcode=int(result.retcode),
        comment=str(result),
    )


def evaluate_and_execute_demo(
    mt5,
    symbol: str,
    raw_df: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    volume: float,
    deviation: int,
    stop_atr: float = 1.2,
    take_profit_atr: float = 0.8,
    max_hold_minutes: float | None = None,
    allow_short: bool = False,
    state: dict | None = None,
    trade_current_signal: bool = False,
    dry_run: bool = False,
    debug_signals: bool = False,
) -> DemoTradeDecision:
    assert_demo_account(mt5)
    ensure_symbol(mt5, symbol)

    df = add_common_indicators(raw_df)
    df = strategy.generate_signals(df)
    if len(df) < 2:
        return DemoTradeDecision(symbol, "HOLD", "NOT_ENOUGH_DATA", signal="UNKNOWN")

    signal_bar = df.iloc[-2]
    bar_id = _bar_id(signal_bar)
    buy_signal = bool(signal_bar["buy_signal"])
    sell_signal = bool(signal_bar["sell_signal"])
    if debug_signals:
        print_signal_debug(symbol, bar_id, signal_bar)
    symbol_state = _symbol_state(state, symbol)
    first_observation = symbol_state is not None and "last_seen_bar" not in symbol_state
    buy_already_traded = symbol_state is not None and symbol_state.get("last_buy_bar") == bar_id
    short_already_traded = symbol_state is not None and symbol_state.get("last_short_bar") == bar_id
    positions = open_positions(mt5, symbol)

    def remember(decision: DemoTradeDecision, has_position: bool | None = None) -> DemoTradeDecision:
        return _remember_signal_state(
            symbol_state,
            bar_id=bar_id,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            has_position=bool(positions) if has_position is None else has_position,
            decision=decision,
        )

    if positions:
        position = positions[0]
        is_short = getattr(position, "type", None) == mt5.POSITION_TYPE_SELL
        should_close_for_signal = buy_signal if is_short else sell_signal
        should_close_for_time = (
            max_hold_minutes is not None
            and max_hold_minutes > 0
            and _position_age_minutes(mt5, position) >= max_hold_minutes
        )

        if should_close_for_signal or should_close_for_time:
            close_reason = "MAX_HOLD_TIME" if should_close_for_time else "OPPOSITE_SIGNAL"
            if dry_run:
                bid, ask = latest_bid_ask(mt5, symbol)
                decision = DemoTradeDecision(
                    symbol,
                    "CLOSE_SIGNAL",
                    f"DRY_RUN_{close_reason}",
                    volume=float(position.volume),
                    price=ask if is_short else bid,
                )
                return remember(_with_position(decision, position))
            decision = close_position(mt5, position, deviation)
            decision.reason = f"{close_reason}: {decision.reason}"
            _with_position(decision, position)
            return remember(decision, has_position=decision.action != "CLOSE_SENT")
        return remember(_with_position(DemoTradeDecision(symbol, "HOLD", "POSITION_OPEN"), position))

    if first_observation and (buy_signal or sell_signal) and not trade_current_signal:
        return remember(
            DemoTradeDecision(symbol, "HOLD", "WARMUP_CURRENT_SIGNAL"),
            has_position=False,
        )

    if buy_signal:
        if buy_already_traded:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "BUY_SIGNAL_ALREADY_TRADED"),
                has_position=False,
            )

        bid, ask = latest_bid_ask(mt5, symbol)
        atr = float(signal_bar["atr"])
        sl = bid - float(stop_atr * atr)
        tp = ask + float(take_profit_atr * atr)
        if dry_run:
            return remember(
                DemoTradeDecision(
                    symbol,
                    "BUY_SIGNAL",
                    "DRY_RUN_BUY_SIGNAL",
                    signal="BUY",
                    volume=normalize_volume(mt5, symbol, volume),
                    price=ask,
                    sl=sl,
                    tp=tp,
                ),
                has_position=False,
            )
        decision = send_market_buy(mt5, symbol, volume, sl, tp, deviation)
        return remember(decision, has_position=decision.action == "BUY_SENT")

    if sell_signal and allow_short:
        if short_already_traded:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "SHORT_SIGNAL_ALREADY_TRADED"),
                has_position=False,
            )

        bid, ask = latest_bid_ask(mt5, symbol)
        atr = float(signal_bar["atr"])
        sl = ask + float(stop_atr * atr)
        tp = bid - float(take_profit_atr * atr)
        if dry_run:
            return remember(
                DemoTradeDecision(
                    symbol,
                    "SHORT_SIGNAL",
                    "DRY_RUN_SHORT_SIGNAL",
                    signal="SELL",
                    volume=normalize_volume(mt5, symbol, volume),
                    price=bid,
                    sl=sl,
                    tp=tp,
                ),
                has_position=False,
            )
        decision = send_market_sell_short(mt5, symbol, volume, sl, tp, deviation)
        return remember(decision, has_position=decision.action == "SHORT_SENT")

    if not buy_signal:
        reason = "SELL_SIGNAL_NO_POSITION" if sell_signal else "NO_SIGNAL"
        return remember(DemoTradeDecision(symbol, "HOLD", reason), has_position=False)
