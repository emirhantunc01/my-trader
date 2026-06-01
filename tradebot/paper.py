from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import BacktestConfig
from .indicators import add_common_indicators
from .strategies import Strategy


@dataclass
class PaperDecision:
    symbol: str
    action: str
    reason: str
    price: float
    quantity: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state(initial_balance: float) -> dict:
    return {
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "initial_balance": float(initial_balance),
        "cash": float(initial_balance),
        "positions": {},
        "transactions": [],
    }


def load_state(path: Path, initial_balance: float, reset: bool = False) -> dict:
    if reset or not path.exists():
        return default_state(initial_balance)
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("positions", {})
    state.setdefault("transactions", [])
    state.setdefault("cash", float(initial_balance))
    state.setdefault("initial_balance", float(initial_balance))
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now_iso()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def _position_size(
    cash: float,
    equity: float,
    entry_price: float,
    atr: float,
    config: BacktestConfig,
) -> float:
    stop_distance = config.atr_stop_multiplier * atr
    if cash <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0.0
    risk_amount = equity * config.risk_per_trade_pct
    risk_based_size = risk_amount / stop_distance
    affordable_size = (cash * config.target_position_pct) / (
        entry_price * (1 + config.commission)
    )
    return max(0.0, min(risk_based_size, affordable_size))


def portfolio_value(state: dict, market_prices: dict[str, float]) -> float:
    value = float(state["cash"])
    for symbol, position in state["positions"].items():
        price = market_prices.get(symbol, float(position["entry_price"]))
        value += float(position["quantity"]) * price
    return value


def _record_transaction(
    state: dict,
    symbol: str,
    action: str,
    price: float,
    quantity: float,
    reason: str,
    extra: dict | None = None,
) -> None:
    transaction = {
        "timestamp": _now_iso(),
        "symbol": symbol,
        "action": action,
        "price": float(price),
        "quantity": float(quantity),
        "reason": reason,
    }
    if extra:
        transaction.update(extra)
    state["transactions"].append(transaction)


def _buy(
    state: dict,
    symbol: str,
    price: float,
    quantity: float,
    atr: float,
    config: BacktestConfig,
    reason: str,
) -> PaperDecision:
    gross = quantity * price
    commission = gross * config.commission
    state["cash"] = float(state["cash"]) - gross - commission
    state["positions"][symbol] = {
        "quantity": quantity,
        "entry_price": price,
        "entry_commission": commission,
        "entry_time": _now_iso(),
        "trailing_stop": price - config.atr_stop_multiplier * atr,
        "highest_close": price,
    }
    _record_transaction(
        state,
        symbol,
        "BUY",
        price,
        quantity,
        reason,
        {"commission": commission},
    )
    return PaperDecision(symbol=symbol, action="BUY", reason=reason, price=price, quantity=quantity)


def _sell(
    state: dict,
    symbol: str,
    price: float,
    config: BacktestConfig,
    reason: str,
) -> PaperDecision:
    position = state["positions"].pop(symbol)
    quantity = float(position["quantity"])
    sell_value = quantity * price
    sell_commission = sell_value * config.commission
    buy_value = quantity * float(position["entry_price"])
    entry_commission = float(position.get("entry_commission", 0.0))
    pnl = sell_value - sell_commission - buy_value - entry_commission
    state["cash"] = float(state["cash"]) + sell_value - sell_commission
    _record_transaction(
        state,
        symbol,
        "SELL",
        price,
        quantity,
        reason,
        {
            "commission": sell_commission,
            "pnl": pnl,
            "return_pct": pnl / (buy_value + entry_commission) * 100,
        },
    )
    return PaperDecision(symbol=symbol, action="SELL", reason=reason, price=price, quantity=quantity)


def evaluate_symbol_once(
    symbol: str,
    raw_df: pd.DataFrame,
    bid: float,
    ask: float,
    state: dict,
    strategy: Strategy,
    config: BacktestConfig,
) -> PaperDecision:
    df = add_common_indicators(raw_df)
    df = strategy.generate_signals(df)
    if len(df) < 2:
        return PaperDecision(symbol, "HOLD", "NOT_ENOUGH_DATA", bid)

    signal_bar = df.iloc[-2]
    latest_bar = df.iloc[-1]
    symbol = symbol.upper()
    market_prices = {symbol: bid}
    equity = portfolio_value(state, market_prices)
    position = state["positions"].get(symbol)

    if position:
        highest_close = max(float(position.get("highest_close", 0.0)), float(signal_bar["Close"]))
        trailing_stop = max(
            float(position["trailing_stop"]),
            float(signal_bar["Close"] - config.atr_stop_multiplier * signal_bar["atr"]),
        )
        position["highest_close"] = highest_close
        position["trailing_stop"] = trailing_stop

        stop_hit = float(latest_bar["Low"]) <= trailing_stop
        if stop_hit:
            return _sell(state, symbol, bid * (1 - config.slippage), config, "PAPER_ATR_STOP")
        if bool(signal_bar["sell_signal"]):
            return _sell(state, symbol, bid * (1 - config.slippage), config, "PAPER_SELL_SIGNAL")
        return PaperDecision(symbol, "HOLD", "POSITION_OPEN", bid)

    if bool(signal_bar["buy_signal"]):
        entry_price = ask * (1 + config.slippage)
        quantity = _position_size(
            cash=float(state["cash"]),
            equity=equity,
            entry_price=entry_price,
            atr=float(signal_bar["atr"]),
            config=config,
        )
        if quantity <= 0:
            return PaperDecision(symbol, "HOLD", "NO_AVAILABLE_SIZE", ask)
        return _buy(state, symbol, entry_price, quantity, float(signal_bar["atr"]), config, "PAPER_BUY_SIGNAL")

    return PaperDecision(symbol, "HOLD", "NO_SIGNAL", bid)


def paper_summary(state: dict, market_prices: dict[str, float]) -> dict:
    equity = portfolio_value(state, market_prices)
    initial = float(state["initial_balance"])
    return {
        "cash": float(state["cash"]),
        "equity": equity,
        "initial_balance": initial,
        "return_pct": (equity / initial - 1) * 100 if initial else 0.0,
        "open_positions": len(state["positions"]),
        "transactions": len(state["transactions"]),
    }
