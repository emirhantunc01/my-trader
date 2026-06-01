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
        symbol_state["last_sell_bar"] = bar_id
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
    steps = math.floor(((requested - min_volume) / step) + 1e-9)
    volume = min_volume + steps * step
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(volume, precision)


def open_positions(mt5, symbol: str) -> list:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [position for position in positions if getattr(position, "magic", None) == DEMO_MAGIC]


def _position_ticket(position) -> str:
    return str(getattr(position, "ticket", ""))


def _position_side(mt5, position) -> str:
    return "SHORT" if getattr(position, "type", None) == mt5.POSITION_TYPE_SELL else "LONG"


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


def _bars_since(df: pd.DataFrame, prior_bar_id: str | None) -> int | None:
    if not prior_bar_id:
        return None
    current_position = len(df) - 2
    if current_position < 0:
        return None
    for idx_position, idx in enumerate(df.index):
        idx_id = idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        if idx_id == prior_bar_id:
            return current_position - idx_position
    return None


def _cooldown_ready(
    df: pd.DataFrame,
    symbol_state: dict | None,
    key: str,
    cooldown_bars: int,
) -> bool:
    if symbol_state is None or cooldown_bars <= 0:
        return True
    bars = _bars_since(df, symbol_state.get(key))
    return bars is None or bars >= cooldown_bars


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _smart_order_volume(
    mt5,
    symbol: str,
    base_volume: float,
    signal_bar: pd.Series,
    symbol_state: dict | None,
    size_multiplier: float = 1.0,
) -> tuple[float, dict]:
    requested = float(base_volume) * float(size_multiplier)
    atr_std = _safe_float(signal_bar.get("atr_std_20"), default=0.0)
    atr_std_mean = _safe_float(signal_bar.get("atr_std_20_mean"), default=0.0)
    volatility_multiplier = 1.0
    volatility_regime = "NEUTRAL"
    if atr_std > 0 and atr_std_mean > 0:
        if atr_std > atr_std_mean:
            volatility_multiplier = 0.70
            volatility_regime = "HIGH_VOL_REDUCED"
        elif atr_std < atr_std_mean:
            volatility_multiplier = 1.50
            volatility_regime = "LOW_VOL_BOOSTED"

    pnl_multiplier = 1.0
    cumulative_pnl = _safe_float(
        symbol_state.get("cumulative_pnl") if symbol_state is not None else 0.0
    )
    if cumulative_pnl > 0:
        pnl_multiplier = 1.15
    elif cumulative_pnl < 0:
        pnl_multiplier = 0.85

    requested *= volatility_multiplier * pnl_multiplier
    normalized = normalize_volume(mt5, symbol, requested)
    plan = {
        "base_volume": float(base_volume),
        "size_multiplier": float(size_multiplier),
        "volatility_regime": volatility_regime,
        "volatility_multiplier": volatility_multiplier,
        "pnl_multiplier": pnl_multiplier,
        "requested_volume": requested,
        "normalized_volume": normalized,
    }
    if symbol_state is not None:
        symbol_state["last_volume_plan"] = plan
    return normalized, plan


def _position_record(symbol_state: dict | None, ticket: str) -> dict | None:
    if symbol_state is None:
        return None
    for record in symbol_state.setdefault("position_list", []):
        if str(record.get("ticket")) == str(ticket):
            return record
    return None


def _sync_position_state(
    mt5,
    symbol_state: dict | None,
    positions: list,
    bar_id: str,
    bid: float,
    ask: float,
) -> None:
    if symbol_state is None:
        return

    existing = {
        str(record.get("ticket")): record
        for record in symbol_state.setdefault("position_list", [])
    }
    symbol_state.setdefault("cumulative_pnl", 0.0)
    active_records = []
    active_sides = set()
    for position in positions:
        ticket = _position_ticket(position)
        side = _position_side(mt5, position)
        active_sides.add(side)
        entry_price = float(getattr(position, "price_open", 0.0))
        current_volume = float(getattr(position, "volume", 0.0))
        market_price = ask if side == "SHORT" else bid
        record = existing.get(ticket, {})
        record.setdefault("ticket", ticket)
        record.setdefault("entry_bar", bar_id)
        record.setdefault("entry_time", _now_iso())
        record.setdefault("entry_price", entry_price)
        record.setdefault("original_volume", current_volume)
        record.setdefault("partial_30_closed", False)
        record.setdefault("partial_50_closed", False)
        record.setdefault("highest_price", entry_price)
        record.setdefault("lowest_price", entry_price)
        record["side"] = side
        record["current_volume"] = current_volume
        record["last_known_profit"] = float(getattr(position, "profit", 0.0))
        record["last_seen_at"] = _now_iso()
        if side == "LONG":
            record["highest_price"] = max(_safe_float(record.get("highest_price"), entry_price), market_price)
        else:
            low_default = entry_price if _safe_float(record.get("lowest_price"), 0.0) > 0 else market_price
            record["lowest_price"] = min(_safe_float(record.get("lowest_price"), low_default), market_price)
        active_records.append(record)

    for ticket, record in existing.items():
        if ticket not in {_position_ticket(position) for position in positions}:
            closed = dict(record)
            closed["closed_at"] = _now_iso()
            closed["closing_reason"] = "MT5_POSITION_NOT_OPEN"
            if not bool(closed.get("close_pnl_recorded", False)):
                estimated_pnl = _safe_float(closed.get("last_known_profit"), 0.0)
                symbol_state["cumulative_pnl"] = _safe_float(symbol_state.get("cumulative_pnl")) + estimated_pnl
                closed["estimated_realized_pnl"] = estimated_pnl
            symbol_state.setdefault("position_history", []).append(closed)

    symbol_state["position_list"] = active_records
    pyramid = symbol_state.setdefault("pyramid", {})
    for side in ("LONG", "SHORT"):
        if side not in active_sides:
            pyramid.pop(side, None)


def _mark_position_flag(
    symbol_state: dict | None,
    position,
    flag: str,
    value: bool = True,
) -> None:
    record = _position_record(symbol_state, _position_ticket(position))
    if record is not None:
        record[flag] = value


def _record_realized_pnl(
    symbol_state: dict | None,
    position,
    closed_volume: float,
) -> float:
    if symbol_state is None:
        return 0.0
    position_volume = float(getattr(position, "volume", 0.0))
    if position_volume <= 0:
        return 0.0
    realized = float(getattr(position, "profit", 0.0)) * min(float(closed_volume) / position_volume, 1.0)
    symbol_state["cumulative_pnl"] = _safe_float(symbol_state.get("cumulative_pnl")) + realized
    symbol_state["last_realized_pnl"] = realized
    symbol_state["last_realized_at"] = _now_iso()
    return realized


def _close_request_volume(mt5, position, requested_volume: float) -> float:
    symbol = position.symbol
    position_volume = float(position.volume)
    info = mt5.symbol_info(symbol)
    if info is None:
        return min(position_volume, float(requested_volume))
    min_volume = float(info.volume_min)
    if requested_volume >= position_volume:
        return position_volume
    volume = normalize_volume(mt5, symbol, requested_volume)
    if position_volume - volume < min_volume:
        return position_volume
    return min(position_volume, volume)


def _side_positions(mt5, positions: list, side: str) -> list:
    return [position for position in positions if _position_side(mt5, position) == side]


def _side_profit_pct(mt5, positions: list, side: str, bid: float, ask: float) -> float:
    side_positions = _side_positions(mt5, positions, side)
    volume = sum(float(position.volume) for position in side_positions)
    if volume <= 0:
        return 0.0
    entry = sum(float(position.price_open) * float(position.volume) for position in side_positions) / volume
    current = bid if side == "LONG" else ask
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return (current / entry) - 1.0
    return (entry / current) - 1.0


def _summary_decision(symbol: str, decisions: list[DemoTradeDecision]) -> DemoTradeDecision:
    if not decisions:
        return DemoTradeDecision(symbol, "HOLD", "NO_SIGNAL")
    if len(decisions) == 1:
        return decisions[0]
    last = decisions[-1]
    return DemoTradeDecision(
        symbol=symbol,
        action="MANAGED",
        reason="; ".join(f"{decision.action}:{decision.reason}" for decision in decisions),
        signal=last.signal,
        volume=sum(float(decision.volume) for decision in decisions),
        price=last.price,
        sl=last.sl,
        tp=last.tp,
        retcode=last.retcode,
        comment=last.comment,
        position_side=last.position_side,
        position_volume=last.position_volume,
        position_entry=last.position_entry,
        position_profit=last.position_profit,
    )


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


def close_position(
    mt5,
    position,
    deviation: int,
    volume: float | None = None,
    reason: str = "CLOSE",
) -> DemoTradeDecision:
    symbol = position.symbol
    bid, ask = latest_bid_ask(mt5, symbol)
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = bid if order_type == mt5.ORDER_TYPE_SELL else ask
    close_volume = float(position.volume) if volume is None else _close_request_volume(mt5, position, volume)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": close_volume,
        "type": order_type,
        "position": int(position.ticket),
        "price": price,
        "deviation": deviation,
        "magic": DEMO_MAGIC,
        "comment": f"my-trader demo {reason.lower()}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, symbol),
    }
    result = _send_with_filling_fallbacks(mt5, request)
    if result is None:
        return DemoTradeDecision(
            symbol=symbol,
            action="CLOSE_FAILED",
            reason=str(mt5.last_error()),
            volume=close_volume,
            price=price,
        )
    return DemoTradeDecision(
        symbol=symbol,
        action="CLOSE_SENT" if result.retcode == mt5.TRADE_RETCODE_DONE else "CLOSE_FAILED",
        reason=getattr(result, "comment", ""),
        volume=close_volume,
        price=price,
        retcode=int(result.retcode),
        comment=str(result),
    )


def _build_demo_signal_frame(
    raw_df: pd.DataFrame,
    strategy: Strategy,
    confirm_raw_df: pd.DataFrame | None = None,
    signal_raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    primary_df = add_common_indicators(raw_df)
    if hasattr(strategy, "generate_multi_timeframe_signals") and (
        confirm_raw_df is not None or signal_raw_df is not None
    ):
        confirm_df = add_common_indicators(confirm_raw_df) if confirm_raw_df is not None else None
        signal_df = add_common_indicators(signal_raw_df) if signal_raw_df is not None else None
        return strategy.generate_multi_timeframe_signals(primary_df, confirm_df, signal_df)
    return strategy.generate_signals(primary_df)


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
    confirm_raw_df: pd.DataFrame | None = None,
    signal_raw_df: pd.DataFrame | None = None,
    max_positions_per_direction: int = 3,
    pyramid_step_pct: float = 0.005,
    pyramid_add_volume_pct: float = 0.50,
    pyramid_stop_atr: float = 1.5,
    partial_first_profit_pct: float = 0.003,
    partial_second_profit_pct: float = 0.006,
    partial_first_volume_pct: float = 0.30,
    partial_second_volume_pct: float = 0.50,
    trailing_stop_atr: float = 1.2,
    signal_cooldown_bars: int = 2,
    close_all_profit_pct: float | None = 0.01,
) -> DemoTradeDecision:
    assert_demo_account(mt5)
    ensure_symbol(mt5, symbol)

    df = _build_demo_signal_frame(raw_df, strategy, confirm_raw_df, signal_raw_df)
    if len(df) < 2:
        return DemoTradeDecision(symbol, "HOLD", "NOT_ENOUGH_DATA", signal="UNKNOWN")

    signal_bar = df.iloc[-2]
    bar_id = _bar_id(signal_bar)
    buy_signal = bool(signal_bar["buy_signal"])
    sell_signal = bool(signal_bar["sell_signal"])
    if debug_signals:
        print_signal_debug(symbol, bar_id, signal_bar)
    symbol_state = _symbol_state(state, symbol)
    if symbol_state is not None and "last_sell_bar" not in symbol_state:
        if "last_short_bar" in symbol_state:
            symbol_state["last_sell_bar"] = symbol_state["last_short_bar"]
    first_observation = symbol_state is not None and "last_seen_bar" not in symbol_state
    positions = open_positions(mt5, symbol)
    bid, ask = latest_bid_ask(mt5, symbol)
    atr = _safe_float(signal_bar.get("atr"), default=0.0)
    max_positions_per_direction = max(1, int(max_positions_per_direction))
    _sync_position_state(mt5, symbol_state, positions, bar_id, bid, ask)

    def remember(decision: DemoTradeDecision, has_position: bool | None = None) -> DemoTradeDecision:
        return _remember_signal_state(
            symbol_state,
            bar_id=bar_id,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            has_position=bool(positions) if has_position is None else has_position,
            decision=decision,
        )

    managed_decisions: list[DemoTradeDecision] = []
    profit_target_sides = set()
    if close_all_profit_pct is not None and close_all_profit_pct > 0:
        for side in ("LONG", "SHORT"):
            if _side_profit_pct(mt5, positions, side, bid, ask) >= close_all_profit_pct:
                profit_target_sides.add(side)

    for position in list(positions):
        side = _position_side(mt5, position)
        is_short = side == "SHORT"
        current_price = ask if is_short else bid
        entry_price = float(getattr(position, "price_open", 0.0))
        position_volume = float(getattr(position, "volume", 0.0))
        record = _position_record(symbol_state, _position_ticket(position)) or {}
        if entry_price <= 0 or position_volume <= 0:
            continue

        profit_pct = (entry_price / current_price - 1.0) if is_short else (current_price / entry_price - 1.0)
        should_close_for_signal = buy_signal if is_short else sell_signal
        should_close_for_time = (
            max_hold_minutes is not None
            and max_hold_minutes > 0
            and _position_age_minutes(mt5, position) >= max_hold_minutes
        )
        trailing_hit = False
        if atr > 0:
            if is_short:
                low_watermark = _safe_float(record.get("lowest_price"), entry_price)
                trailing_hit = current_price >= low_watermark + trailing_stop_atr * atr
            else:
                high_watermark = _safe_float(record.get("highest_price"), entry_price)
                trailing_hit = current_price <= high_watermark - trailing_stop_atr * atr

        close_reason = ""
        if should_close_for_signal:
            close_reason = "OPPOSITE_SIGNAL"
        elif should_close_for_time:
            close_reason = "MAX_HOLD_TIME"
        elif side in profit_target_sides:
            close_reason = "SYMBOL_PROFIT_TARGET"
        elif trailing_hit:
            close_reason = "TRAILING_STOP"

        if close_reason:
            if dry_run:
                decision = DemoTradeDecision(
                    symbol,
                    "CLOSE_SIGNAL",
                    f"DRY_RUN_{close_reason}",
                    volume=position_volume,
                    price=current_price,
                )
            else:
                decision = close_position(mt5, position, deviation, reason=close_reason)
                if decision.action == "CLOSE_SENT":
                    _record_realized_pnl(symbol_state, position, decision.volume)
                    _mark_position_flag(symbol_state, position, "close_pnl_recorded")
            decision.reason = f"{close_reason}: {decision.reason}"
            managed_decisions.append(_with_position(decision, position))
            continue

        original_volume = _safe_float(record.get("original_volume"), position_volume)
        partial_action = ""
        partial_volume = 0.0
        partial_flag = ""
        if (
            profit_pct >= partial_first_profit_pct
            and partial_first_volume_pct > 0
            and not bool(record.get("partial_30_closed", False))
        ):
            partial_action = "PARTIAL_30"
            partial_volume = original_volume * partial_first_volume_pct
            partial_flag = "partial_30_closed"
        elif (
            profit_pct >= partial_second_profit_pct
            and partial_second_volume_pct > 0
            and not bool(record.get("partial_50_closed", False))
        ):
            partial_action = "PARTIAL_50"
            partial_volume = original_volume * partial_second_volume_pct
            partial_flag = "partial_50_closed"

        if partial_action and partial_volume > 0:
            if dry_run:
                close_volume = _close_request_volume(mt5, position, partial_volume)
                decision = DemoTradeDecision(
                    symbol,
                    "PARTIAL_CLOSE_SIGNAL",
                    f"DRY_RUN_{partial_action}",
                    volume=close_volume,
                    price=current_price,
                )
                _mark_position_flag(symbol_state, position, partial_flag)
            else:
                decision = close_position(
                    mt5,
                    position,
                    deviation,
                    volume=partial_volume,
                    reason=partial_action,
                )
                if decision.action == "CLOSE_SENT":
                    decision.action = "PARTIAL_CLOSE_SENT"
                    _mark_position_flag(symbol_state, position, partial_flag)
                    _record_realized_pnl(symbol_state, position, decision.volume)
                    if decision.volume >= position_volume:
                        _mark_position_flag(symbol_state, position, "close_pnl_recorded")
            decision.reason = f"{partial_action}: {decision.reason}"
            managed_decisions.append(_with_position(decision, position))

    if managed_decisions:
        if dry_run:
            return remember(_summary_decision(symbol, managed_decisions), has_position=bool(positions))
        positions = open_positions(mt5, symbol)
        bid, ask = latest_bid_ask(mt5, symbol)
        _sync_position_state(mt5, symbol_state, positions, bar_id, bid, ask)

    if first_observation and (buy_signal or sell_signal) and not trade_current_signal and not positions:
        return remember(
            DemoTradeDecision(symbol, "HOLD", "WARMUP_CURRENT_SIGNAL"),
            has_position=False,
        )

    long_positions = _side_positions(mt5, positions, "LONG")
    short_positions = _side_positions(mt5, positions, "SHORT")
    buy_cooldown_ready = _cooldown_ready(
        df,
        symbol_state,
        "last_buy_bar",
        signal_cooldown_bars,
    )
    sell_cooldown_ready = _cooldown_ready(
        df,
        symbol_state,
        "last_sell_bar",
        signal_cooldown_bars,
    )

    def pyramid_base(side: str, side_positions: list) -> tuple[float, float]:
        pyramid_state = symbol_state.setdefault("pyramid", {}).setdefault(side, {}) if symbol_state is not None else {}
        base_entry = _safe_float(pyramid_state.get("base_entry"), 0.0)
        base_volume = _safe_float(pyramid_state.get("base_volume"), 0.0)
        if base_entry > 0 and base_volume > 0:
            return base_entry, base_volume
        ordered = sorted(side_positions, key=lambda item: int(getattr(item, "time", 0)))
        if not ordered:
            return 0.0, 0.0
        base_entry = float(getattr(ordered[0], "price_open", 0.0))
        base_volume = float(getattr(ordered[0], "volume", 0.0))
        if symbol_state is not None:
            pyramid_state["base_entry"] = base_entry
            pyramid_state["base_volume"] = base_volume
            pyramid_state.setdefault("base_bar", bar_id)
        return base_entry, base_volume

    def store_entry_state(side: str, decision: DemoTradeDecision, was_empty: bool) -> None:
        if symbol_state is None or decision.action not in {"BUY_SENT", "SHORT_SENT"}:
            return
        side_state = symbol_state.setdefault("pyramid", {}).setdefault(side, {})
        if was_empty or _safe_float(side_state.get("base_entry"), 0.0) <= 0:
            side_state["base_entry"] = decision.price
            side_state["base_volume"] = decision.volume
            side_state["base_bar"] = bar_id
        side_state["last_entry_bar"] = bar_id
        side_state["last_entry_volume"] = decision.volume

    if buy_signal:
        if short_positions:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "SHORT_POSITION_STILL_OPEN"),
                has_position=True,
            )
        if len(long_positions) >= max_positions_per_direction:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "LONG_POSITION_LIMIT"),
                has_position=bool(positions),
            )
        if not buy_cooldown_ready:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "BUY_SIGNAL_COOLDOWN"),
                has_position=bool(positions),
            )

        entry_reason = "BUY_SIGNAL"
        size_multiplier = 1.0
        order_stop_atr = stop_atr
        was_empty = len(long_positions) == 0
        if long_positions:
            base_entry, base_volume = pyramid_base("LONG", long_positions)
            next_level = len(long_positions)
            threshold = float(pyramid_step_pct) * next_level
            if base_entry <= 0 or bid < base_entry * (1.0 + threshold):
                if managed_decisions:
                    return remember(_summary_decision(symbol, managed_decisions), has_position=True)
                return remember(DemoTradeDecision(symbol, "HOLD", "PYRAMID_LONG_WAIT"), has_position=True)
            entry_reason = f"PYRAMID_LONG_{next_level}"
            size_multiplier = (base_volume / float(volume)) * float(pyramid_add_volume_pct) if volume > 0 else 0.0
            order_stop_atr = pyramid_stop_atr

        order_volume, _ = _smart_order_volume(mt5, symbol, volume, signal_bar, symbol_state, size_multiplier)
        sl = bid - float(order_stop_atr * atr)
        tp = ask + float(take_profit_atr * atr)
        if dry_run:
            return remember(
                DemoTradeDecision(
                    symbol,
                    "BUY_SIGNAL",
                    f"DRY_RUN_{entry_reason}",
                    signal="BUY",
                    volume=order_volume,
                    price=ask,
                    sl=sl,
                    tp=tp,
                ),
                has_position=bool(positions),
            )
        decision = send_market_buy(mt5, symbol, order_volume, sl, tp, deviation)
        decision.reason = f"{entry_reason}: {decision.reason}"
        store_entry_state("LONG", decision, was_empty)
        if decision.action == "BUY_SENT":
            positions = open_positions(mt5, symbol)
            bid, ask = latest_bid_ask(mt5, symbol)
            _sync_position_state(mt5, symbol_state, positions, bar_id, bid, ask)
        if managed_decisions:
            decision.reason = f"{'; '.join(item.reason for item in managed_decisions)}; {decision.reason}"
        return remember(decision, has_position=bool(positions) or decision.action == "BUY_SENT")

    if sell_signal and allow_short:
        if long_positions:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "LONG_POSITION_STILL_OPEN"),
                has_position=True,
            )
        if len(short_positions) >= max_positions_per_direction:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "SHORT_POSITION_LIMIT"),
                has_position=bool(positions),
            )
        if not sell_cooldown_ready:
            return remember(
                DemoTradeDecision(symbol, "HOLD", "SELL_SIGNAL_COOLDOWN"),
                has_position=bool(positions),
            )

        entry_reason = "SHORT_SIGNAL"
        size_multiplier = 1.0
        order_stop_atr = stop_atr
        was_empty = len(short_positions) == 0
        if short_positions:
            base_entry, base_volume = pyramid_base("SHORT", short_positions)
            next_level = len(short_positions)
            threshold = float(pyramid_step_pct) * next_level
            if base_entry <= 0 or ask > base_entry * (1.0 - threshold):
                if managed_decisions:
                    return remember(_summary_decision(symbol, managed_decisions), has_position=True)
                return remember(DemoTradeDecision(symbol, "HOLD", "PYRAMID_SHORT_WAIT"), has_position=True)
            entry_reason = f"PYRAMID_SHORT_{next_level}"
            size_multiplier = (base_volume / float(volume)) * float(pyramid_add_volume_pct) if volume > 0 else 0.0
            order_stop_atr = pyramid_stop_atr

        order_volume, _ = _smart_order_volume(mt5, symbol, volume, signal_bar, symbol_state, size_multiplier)
        sl = ask + float(order_stop_atr * atr)
        tp = bid - float(take_profit_atr * atr)
        if dry_run:
            return remember(
                DemoTradeDecision(
                    symbol,
                    "SHORT_SIGNAL",
                    f"DRY_RUN_{entry_reason}",
                    signal="SELL",
                    volume=order_volume,
                    price=bid,
                    sl=sl,
                    tp=tp,
                ),
                has_position=bool(positions),
            )
        decision = send_market_sell_short(mt5, symbol, order_volume, sl, tp, deviation)
        decision.reason = f"{entry_reason}: {decision.reason}"
        store_entry_state("SHORT", decision, was_empty)
        if decision.action == "SHORT_SENT":
            positions = open_positions(mt5, symbol)
            bid, ask = latest_bid_ask(mt5, symbol)
            _sync_position_state(mt5, symbol_state, positions, bar_id, bid, ask)
        if managed_decisions:
            decision.reason = f"{'; '.join(item.reason for item in managed_decisions)}; {decision.reason}"
        return remember(decision, has_position=bool(positions) or decision.action == "SHORT_SENT")

    if managed_decisions:
        return remember(_summary_decision(symbol, managed_decisions), has_position=bool(positions))

    if positions:
        side = "MIXED"
        if long_positions and not short_positions:
            side = "LONG"
        elif short_positions and not long_positions:
            side = "SHORT"
        total_volume = sum(float(position.volume) for position in positions)
        total_profit = sum(float(getattr(position, "profit", 0.0)) for position in positions)
        weighted_entry = (
            sum(float(position.price_open) * float(position.volume) for position in positions) / total_volume
            if total_volume > 0
            else 0.0
        )
        decision = DemoTradeDecision(
            symbol,
            "HOLD",
            "POSITION_OPEN",
            position_side=side,
            position_volume=total_volume,
            position_entry=weighted_entry,
            position_profit=total_profit,
        )
        return remember(decision, has_position=True)

    if not buy_signal:
        reason = "SELL_SIGNAL_SHORT_DISABLED" if sell_signal and not allow_short else "NO_SIGNAL"
        return remember(DemoTradeDecision(symbol, "HOLD", reason), has_position=False)

    return remember(DemoTradeDecision(symbol, "HOLD", "NO_ACTION"), has_position=bool(positions))
