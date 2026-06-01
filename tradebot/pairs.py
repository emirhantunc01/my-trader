from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .risk import calculate_metrics


@dataclass
class PairBacktestResult:
    equity: pd.Series
    trades: list[dict]
    metrics: dict[str, float]
    spread: pd.DataFrame
    diagnostics: dict[str, float | str]


def pair_diagnostics(
    aligned: pd.DataFrame,
    spread_frame: pd.DataFrame,
    entry_z: float,
) -> dict[str, float | str]:
    returns = aligned[["close_a", "close_b"]].pct_change().dropna()
    correlation = float(returns["close_a"].corr(returns["close_b"]))
    zscore = spread_frame["zscore"].dropna()
    z_crossings = int(((zscore.shift(1) * zscore) < 0).sum())
    entry_events = int((zscore.abs() >= entry_z).sum())
    mean_abs_z = float(zscore.abs().mean()) if len(zscore) else 0.0

    if correlation >= 0.75 and z_crossings >= 8 and entry_events >= 5:
        verdict = "usable"
    elif correlation >= 0.50 and z_crossings >= 4 and entry_events >= 3:
        verdict = "borderline"
    else:
        verdict = "weak"

    return {
        "return_correlation": correlation,
        "zscore_crossings": float(z_crossings),
        "entry_events": float(entry_events),
        "mean_abs_zscore": mean_abs_z,
        "verdict": verdict,
    }


def _slipped(price: float, signed_share_delta: float, slippage: float) -> float:
    if signed_share_delta > 0:
        return price * (1 + slippage)
    return price * (1 - slippage)


def run_pairs_backtest(
    symbol_a: str,
    symbol_b: str,
    data: dict[str, pd.DataFrame],
    config: BacktestConfig,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.50,
) -> PairBacktestResult:
    a = data[symbol_a.upper()]
    b = data[symbol_b.upper()]
    aligned = pd.concat(
        [
            a[["Open", "Close"]].rename(columns={"Open": "open_a", "Close": "close_a"}),
            b[["Open", "Close"]].rename(columns={"Open": "open_b", "Close": "close_b"}),
        ],
        axis=1,
    ).dropna()

    log_a = np.log(aligned["close_a"])
    log_b = np.log(aligned["close_b"])
    beta = log_a.rolling(lookback).cov(log_b) / log_b.rolling(lookback).var()
    spread = log_a - beta * log_b
    zscore = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    spread_frame = pd.DataFrame({"beta": beta, "spread": spread, "zscore": zscore}).dropna()
    if spread_frame.empty:
        raise ValueError(
            "Not enough pair data after lookback calculation. "
            "Use a longer period or a smaller --pair-lookback."
        )
    diagnostics = pair_diagnostics(aligned, spread_frame, entry_z)

    cash = config.initial_capital
    shares_a = 0.0
    shares_b = 0.0
    entry_date = None
    entry_equity = cash
    direction = 0
    trades: list[dict] = []
    equity_curve = [(spread_frame.index[0], config.initial_capital)]

    for i in range(0, len(spread_frame) - 1):
        execution_date = spread_frame.index[i + 1]
        today_z = float(spread_frame["zscore"].iloc[i])
        row = aligned.loc[execution_date]

        if direction != 0 and abs(today_z) <= exit_z:
            exit_a = _slipped(float(row["open_a"]), -shares_a, config.slippage)
            exit_b = _slipped(float(row["open_b"]), -shares_b, config.slippage)
            cash += shares_a * exit_a + shares_b * exit_b
            cash -= (abs(shares_a * exit_a) + abs(shares_b * exit_b)) * config.commission
            pnl = cash - entry_equity
            trades.append(
                {
                    "symbol": f"{symbol_a}/{symbol_b}",
                    "leg": "PAIR",
                    "entry_date": entry_date,
                    "exit_date": execution_date,
                    "entry_price": 0.0,
                    "exit_price": 0.0,
                    "shares": 0.0,
                    "pnl": pnl,
                    "return_pct": pnl / entry_equity * 100,
                    "reason": "Z_EXIT",
                }
            )
            shares_a = 0.0
            shares_b = 0.0
            direction = 0

        if direction == 0 and abs(today_z) >= entry_z:
            direction = -1 if today_z > 0 else 1
            entry_date = execution_date
            entry_equity = cash
            gross_notional = cash * config.target_position_pct
            leg_notional = gross_notional / 2

            signed_a = direction * leg_notional / float(row["open_a"])
            signed_b = -direction * leg_notional / float(row["open_b"])
            entry_a = _slipped(float(row["open_a"]), signed_a, config.slippage)
            entry_b = _slipped(float(row["open_b"]), signed_b, config.slippage)
            shares_a = signed_a
            shares_b = signed_b
            cash -= shares_a * entry_a + shares_b * entry_b
            cash -= (abs(shares_a * entry_a) + abs(shares_b * entry_b)) * config.commission

        close_row = aligned.loc[execution_date]
        equity = cash + shares_a * float(close_row["close_a"]) + shares_b * float(
            close_row["close_b"]
        )
        equity_curve.append((execution_date, equity))

    if direction != 0:
        final_date = spread_frame.index[-1]
        row = aligned.loc[final_date]
        exit_a = _slipped(float(row["close_a"]), -shares_a, config.slippage)
        exit_b = _slipped(float(row["close_b"]), -shares_b, config.slippage)
        cash += shares_a * exit_a + shares_b * exit_b
        cash -= (abs(shares_a * exit_a) + abs(shares_b * exit_b)) * config.commission
        pnl = cash - entry_equity
        trades.append(
            {
                "symbol": f"{symbol_a}/{symbol_b}",
                "leg": "PAIR",
                "entry_date": entry_date,
                "exit_date": final_date,
                "entry_price": 0.0,
                "exit_price": 0.0,
                "shares": 0.0,
                "pnl": pnl,
                "return_pct": pnl / entry_equity * 100,
                "reason": "FINAL_CLOSE",
            }
        )
        equity_curve[-1] = (final_date, cash)

    equity = pd.Series(
        [value for _, value in equity_curve],
        index=[date for date, _ in equity_curve],
        name="pair_equity",
    )
    metrics = calculate_metrics(equity, trades)
    return PairBacktestResult(
        equity=equity,
        trades=trades,
        metrics=metrics,
        spread=spread_frame,
        diagnostics=diagnostics,
    )
