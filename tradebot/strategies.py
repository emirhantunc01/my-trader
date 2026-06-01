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


class FastTrendStrategy(Strategy):
    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config=config, name="fast-trend")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        volume_ok = ~df["volume_pump"]

        df["signal_score"] = (
            (df["Close"] > df["ema_200"]).astype(int)
            + (df["ema_8"] > df["ema_21"]).astype(int)
            + (df["macd_hist"] > 0).astype(int)
        )

        df["buy_signal"] = (df["signal_score"] >= 2) & volume_ok
        df["sell_signal"] = (
            (df["Close"] < df["ema_200"])
            | ((df["ema_8"] < df["ema_21"]) & (df["macd_hist"] < 0))
        )
        return df


class ScalpStrategy(Strategy):
    def __init__(self, config: BacktestConfig) -> None:
        super().__init__(config=config, name="scalp")

    @staticmethod
    def _add_scalp_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr_std_20"] = df["atr"].rolling(20, min_periods=5).std()
        df["atr_std_20_mean"] = df["atr_std_20"].rolling(20, min_periods=5).mean()
        df["m1_uptrend"] = (df["Close"] > df["ema_8"]) & (df["ema_8"] > df["ema_21"])
        df["m1_downtrend"] = (df["Close"] < df["ema_8"]) & (df["ema_8"] < df["ema_21"])
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_scalp_metrics(df)
        volume_ok = ~df["volume_pump"]

        df["long_score"] = (
            (df["ema_8"] > df["ema_21"]).astype(int)
            + (df["Close"] > df["ema_20"]).astype(int)
            + (df["macd_hist"] > 0).astype(int)
            + ((df["rsi"] > 45) & (df["rsi"] < 78)).astype(int)
        )
        df["short_score"] = (
            (df["ema_8"] < df["ema_21"]).astype(int)
            + (df["Close"] < df["ema_20"]).astype(int)
            + (df["macd_hist"] < 0).astype(int)
            + ((df["rsi"] > 22) & (df["rsi"] < 55)).astype(int)
        )
        df["signal_score"] = df["long_score"] - df["short_score"]
        df["buy_signal"] = (df["long_score"] >= 3) & volume_ok
        df["sell_signal"] = (df["short_score"] >= 3) & volume_ok
        return df

    def generate_multi_timeframe_signals(
        self,
        primary_df: pd.DataFrame,
        confirm_df: pd.DataFrame | None = None,
        signal_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        primary = self.generate_signals(primary_df).sort_index()
        primary["m1_buy_signal"] = primary["buy_signal"]
        primary["m1_sell_signal"] = primary["sell_signal"]

        if confirm_df is not None and not confirm_df.empty:
            confirm = self._add_scalp_metrics(confirm_df).sort_index()
            confirm["m5_uptrend"] = (confirm["Close"] > confirm["ema_8"]) & (
                confirm["ema_8"] > confirm["ema_21"]
            )
            confirm["m5_downtrend"] = (confirm["Close"] < confirm["ema_8"]) & (
                confirm["ema_8"] < confirm["ema_21"]
            )
            primary = pd.merge_asof(
                primary,
                confirm[["m5_uptrend", "m5_downtrend"]],
                left_index=True,
                right_index=True,
                direction="backward",
            )
        else:
            primary["m5_uptrend"] = primary["m1_uptrend"]
            primary["m5_downtrend"] = primary["m1_downtrend"]

        if signal_df is not None and not signal_df.empty:
            signal = self.generate_signals(signal_df).sort_index()
            signal = signal.rename(
                columns={
                    "buy_signal": "m15_buy_signal",
                    "sell_signal": "m15_sell_signal",
                    "signal_score": "m15_signal_score",
                }
            )
            primary = pd.merge_asof(
                primary,
                signal[["m15_buy_signal", "m15_sell_signal", "m15_signal_score"]],
                left_index=True,
                right_index=True,
                direction="backward",
            )
        else:
            primary["m15_buy_signal"] = primary["m1_buy_signal"]
            primary["m15_sell_signal"] = primary["m1_sell_signal"]
            primary["m15_signal_score"] = primary["signal_score"]

        volume_ok = ~primary["volume_pump"].fillna(False)
        m1_long_ok = (primary["long_score"] >= 2) & volume_ok
        m1_short_ok = (primary["short_score"] >= 2) & volume_ok
        primary["buy_signal"] = (
            primary["m15_buy_signal"].fillna(False)
            & primary["m5_uptrend"].fillna(False)
            & m1_long_ok
        )
        primary["sell_signal"] = (
            primary["m15_sell_signal"].fillna(False)
            & primary["m5_downtrend"].fillna(False)
            & m1_short_ok
        )
        primary["signal_score"] = (
            primary["long_score"]
            - primary["short_score"]
            + primary["m15_signal_score"].fillna(0)
        )
        return primary


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
    if name == "fast-trend":
        return FastTrendStrategy(config), 0.0
    if name == "scalp":
        return ScalpStrategy(config), 0.0
    if name == "mean-reversion":
        return MeanReversionStrategy(config), 0.0
    if name == "hybrid":
        return HybridStrategy(config), 0.0
    if name == "core-satellite":
        return TrendFollowingStrategy(config), config.core_position_pct
    if name == "pairs":
        raise ValueError("Pairs strategy is handled by the pair backtester.")
    raise ValueError(f"Unknown strategy: {config.strategy}")
