from __future__ import annotations

import math
from dataclasses import dataclass

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
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    retcode: int | None = None
    comment: str = ""


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
        )
    return DemoTradeDecision(
        symbol=symbol,
        action="BUY_SENT" if result.retcode == mt5.TRADE_RETCODE_DONE else "BUY_FAILED",
        reason=getattr(result, "comment", ""),
        volume=volume,
        price=ask,
        sl=request["sl"],
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
) -> DemoTradeDecision:
    assert_demo_account(mt5)
    ensure_symbol(mt5, symbol)

    df = add_common_indicators(raw_df)
    df = strategy.generate_signals(df)
    if len(df) < 2:
        return DemoTradeDecision(symbol, "HOLD", "NOT_ENOUGH_DATA")

    signal_bar = df.iloc[-2]
    positions = open_positions(mt5, symbol)

    if positions and bool(signal_bar["sell_signal"]):
        return close_position(mt5, positions[0], deviation)

    if positions:
        return DemoTradeDecision(symbol, "HOLD", "POSITION_OPEN")

    if not bool(signal_bar["buy_signal"]):
        return DemoTradeDecision(symbol, "HOLD", "NO_SIGNAL")

    bid, ask = latest_bid_ask(mt5, symbol)
    sl = bid - float(config.atr_stop_multiplier * signal_bar["atr"])
    return send_market_buy(mt5, symbol, volume, sl, deviation)
