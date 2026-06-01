from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M2": "TIMEFRAME_M2",
    "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4",
    "M5": "TIMEFRAME_M5",
    "M6": "TIMEFRAME_M6",
    "M10": "TIMEFRAME_M10",
    "M12": "TIMEFRAME_M12",
    "M15": "TIMEFRAME_M15",
    "M20": "TIMEFRAME_M20",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H2": "TIMEFRAME_H2",
    "H3": "TIMEFRAME_H3",
    "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6",
    "H8": "TIMEFRAME_H8",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


def import_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 Python package is not installed. "
            "Install it with: python -m pip install MetaTrader5"
        ) from exc
    return mt5


def resolve_timeframe(mt5, timeframe: str) -> int:
    key = timeframe.upper()
    if key not in TIMEFRAME_MAP:
        allowed = ", ".join(TIMEFRAME_MAP)
        raise ValueError(f"Unsupported MT5 timeframe '{timeframe}'. Use one of: {allowed}")
    return getattr(mt5, TIMEFRAME_MAP[key])


@contextmanager
def mt5_session(
    terminal_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    timeout: int = 60_000,
) -> Iterator:
    mt5 = import_mt5()

    kwargs = {"timeout": timeout}
    if terminal_path:
        kwargs["path"] = terminal_path
    if login is not None:
        kwargs["login"] = login
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        yield mt5
    finally:
        mt5.shutdown()


def account_snapshot(mt5) -> dict:
    info = mt5.account_info()
    if info is None:
        return {"connected": False, "last_error": str(mt5.last_error())}
    data = info._asdict()
    return {
        "connected": True,
        "login": data.get("login"),
        "server": data.get("server"),
        "currency": data.get("currency"),
        "balance": data.get("balance"),
        "equity": data.get("equity"),
        "leverage": data.get("leverage"),
    }


def ensure_symbol(mt5, symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"MT5 symbol not found: {symbol}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select MT5 symbol: {symbol}")


def fetch_rates(mt5, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    ensure_symbol(mt5, symbol)
    mt5_timeframe = resolve_timeframe(mt5, timeframe)
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed for {symbol}: {mt5.last_error()}")
    if len(rates) == 0:
        raise RuntimeError(f"MT5 returned no rates for {symbol}.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        },
        inplace=True,
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Symbol"] = symbol
    return df


def fetch_universe_rates(
    mt5,
    symbols: tuple[str, ...],
    timeframe: str,
    bars: int,
) -> dict[str, pd.DataFrame]:
    return {symbol: fetch_rates(mt5, symbol, timeframe, bars) for symbol in symbols}


def fetch_timeframe_rates(
    mt5,
    symbol: str,
    timeframes: dict[str, str],
    bars: int | dict[str, int],
) -> dict[str, pd.DataFrame]:
    frames = {}
    for label, timeframe in timeframes.items():
        timeframe_bars = bars[label] if isinstance(bars, dict) else bars
        frames[label] = fetch_rates(mt5, symbol, timeframe, int(timeframe_bars))
    return frames


def fetch_universe_timeframe_rates(
    mt5,
    symbols: tuple[str, ...],
    timeframes: dict[str, str],
    bars: int | dict[str, int],
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        symbol: fetch_timeframe_rates(mt5, symbol, timeframes, bars)
        for symbol in symbols
    }


def latest_bid_ask(mt5, symbol: str) -> tuple[float, float]:
    ensure_symbol(mt5, symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick failed for {symbol}: {mt5.last_error()}")
    return float(tick.bid), float(tick.ask)
