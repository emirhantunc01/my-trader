from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BacktestConfig


@dataclass
class Strategy:
    config: BacktestConfig
    name: str

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TrendFollowingStrategy(Strategy):
    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config=config, name="trend")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        volume_ok = ~df["volume_pump"]

        df["signal_score"] = (
            (df["Close"] > df["ema_200"]).astype(int)
            + (df["ema_8"] > df["ema_21"]).astype(int)
            + (df["ema_20"] > df["ema_50"]).astype(int)
            + (df["adx"] > self.config.adx_threshold).astype(int)
            + (df["macd_hist"] > 0).astype(int)
            + ((df["rsi"] > 35) & (df["rsi"] < 82)).astype(int)
            + (df["cmf"] > -0.10).astype(int)
        )

        df["buy_signal"] = (df["signal_score"] >= 5) & volume_ok
        df["sell_signal"] = (
            (df["Close"] < df["ema_200"])
            | ((df["ema_8"] < df["ema_21"]) & (df["macd_hist"] < 0))
            | ((df["Close"] < df["ema_50"]) & (df["cmf"] < -0.20))
        )
        return df


class MeanReversionStrategy(Strategy):
    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config=config, name="mean-reversion")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        not_crashing = df["Close"] > df["ema_200"] * 0.90
        volume_ok = ~df["volume_pump"]

        df["signal_score"] = (
            (df["Close"] < df["bb_low"]).astype(int)
            + (df["zscore_20"] < -1.75).astype(int)
            + (df["rsi"] < 40).astype(int)
            + (df["mfi"] < 45).astype(int)
            + (df["cmf"] > -0.35).astype(int)
        )

        df["buy_signal"] = (df["signal_score"] >= 3) & not_crashing & volume_ok
        df["sell_signal"] = (
            (df["Close"] > df["bb_mid"])
            | (df["zscore_20"] > -0.10)
            | (df["rsi"] > 62)
            | (df["Close"] < df["ema_200"] * 0.82)
        )
        return df


class HybridStrategy(Strategy):
    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config=config, name="hybrid")
        self.trend = TrendFollowingStrategy(config)
        self.mean_reversion = MeanReversionStrategy(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = self.trend.generate_signals(df)
        mean_reversion = self.mean_reversion.generate_signals(df)

        out = df.copy()
        out["signal_score"] = trend["signal_score"].clip(upper=5) + mean_reversion[
            "signal_score"
        ].clip(upper=3)
        out["buy_signal"] = trend["buy_signal"] | mean_reversion["buy_signal"]
        out["sell_signal"] = trend["sell_signal"] & mean_reversion["sell_signal"]
        return out


def strategy_factory(config: BacktestConfig) -> tuple[Strategy, float]:
    name = config.strategy.lower().strip()
    if name == "trend":
        return TrendFollowingStrategy(config), 0.0
    if name == "mean-reversion":
        return MeanReversionStrategy(config), 0.0
    if name == "hybrid":
        return HybridStrategy(config), 0.0
    if name == "core-satellite":
        return TrendFollowingStrategy(config), config.core_position_pct
    if name == "pairs":
        raise ValueError("Pairs strategy is handled by the pair backtester.")
    raise ValueError(f"Unknown strategy: {config.strategy}")
