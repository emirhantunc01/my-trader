import numpy as np
import pandas as pd
import ta


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["ema_8"] = ta.trend.EMAIndicator(close, window=8).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(close, window=100).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["sma_20"] = close.rolling(20).mean()

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["adx"] = adx.adx()

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr"] = atr.average_true_range()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    df["mfi"] = ta.volume.MFIIndicator(
        high=high,
        low=low,
        close=close,
        volume=volume,
        window=14,
    ).money_flow_index()
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
        high=high,
        low=low,
        close=close,
        volume=volume,
        window=20,
    ).chaikin_money_flow()

    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_ma30"] = volume.rolling(30).mean()
    df["volume_pump"] = volume > df["volume_ma30"] * 20
    df["zscore_20"] = rolling_zscore(close, 20)

    return df.dropna().copy()
