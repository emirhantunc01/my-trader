from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).resolve().parents[1] / ".yfinance_cache"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    first_level = set(df.columns.get_level_values(0))
    if set(REQUIRED_COLUMNS).issubset(first_level):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        return df

    second_level = set(df.columns.get_level_values(1))
    if set(REQUIRED_COLUMNS).issubset(second_level):
        df = df.copy()
        df.columns = df.columns.get_level_values(1)
        return df

    raise ValueError("Could not understand yfinance MultiIndex columns.")


def download_ohlcv(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR))

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    df = _flatten_yfinance_columns(df)

    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}.")

    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{symbol} data is missing columns: {', '.join(sorted(missing))}")

    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()
    df["Symbol"] = symbol.upper()

    return df


def download_universe(
    symbols: tuple[str, ...],
    period: str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    return {
        symbol.upper(): download_ohlcv(symbol.upper(), period=period, interval=interval)
        for symbol in symbols
    }
