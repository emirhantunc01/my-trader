from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd_pct = float(drawdown.min() * 100)
    max_dd_dollars = float((equity - running_max).min())
    return max_dd_pct, max_dd_dollars


def _years(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    return days / 365.25


def profit_factor(trades: Iterable[dict]) -> float:
    trade_list = list(trades)
    gross_profit = sum(float(trade["pnl"]) for trade in trade_list if trade["pnl"] > 0)
    gross_loss = abs(sum(float(trade["pnl"]) for trade in trade_list if trade["pnl"] < 0))
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_metrics(
    equity: pd.Series,
    trades: list[dict] | None = None,
    benchmark_equity: pd.Series | None = None,
    market_exposure: float | None = None,
) -> dict[str, float]:
    trades = trades or []
    equity = equity.dropna()
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = (final / initial - 1) * 100

    years = _years(equity)
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    returns = equity.pct_change().dropna()
    volatility = float(returns.std() * math.sqrt(252) * 100) if len(returns) > 1 else 0.0
    sharpe = 0.0
    if len(returns) > 1 and returns.std() != 0:
        sharpe = float(math.sqrt(252) * returns.mean() / returns.std())

    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 1 and downside.std() != 0:
        sortino = float(math.sqrt(252) * returns.mean() / downside.std())

    max_dd_pct, max_dd_dollars = max_drawdown(equity)
    calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0
    recovery = (final - initial) / abs(max_dd_dollars) if max_dd_dollars != 0 else 0.0

    winners = [trade for trade in trades if trade.get("pnl", 0) > 0]
    trade_count = len(trades)
    win_rate = len(winners) / trade_count * 100 if trade_count else 0.0
    average_trade = (
        sum(float(trade.get("return_pct", 0)) for trade in trades) / trade_count
        if trade_count
        else 0.0
    )

    metrics = {
        "initial_capital": initial,
        "final_capital": final,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "volatility_pct": volatility,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "recovery_factor": recovery,
        "trade_count": float(trade_count),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor(trades),
        "average_trade_pct": average_trade,
    }

    if market_exposure is not None:
        metrics["market_exposure_pct"] = float(market_exposure)

    if benchmark_equity is not None and not benchmark_equity.empty:
        aligned = pd.concat([equity, benchmark_equity], axis=1).dropna()
        aligned.columns = ["strategy", "benchmark"]
        if not aligned.empty:
            benchmark_return = (
                aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1
            ) * 100
            benchmark_dd, _ = max_drawdown(aligned["benchmark"])
            metrics["benchmark_return_pct"] = float(benchmark_return)
            metrics["benchmark_max_drawdown_pct"] = float(benchmark_dd)
            metrics["alpha_vs_benchmark_pct"] = float(total_return - benchmark_return)
            benchmark_returns = aligned["benchmark"].pct_change().dropna()
            strategy_returns = aligned["strategy"].pct_change().dropna()
            if len(strategy_returns) > 2 and np.var(benchmark_returns) != 0:
                beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(
                    benchmark_returns
                )
                metrics["beta_vs_benchmark"] = float(beta)

    return metrics


def split_metrics(
    equity: pd.Series,
    trades: list[dict],
    train_ratio: float,
    benchmark_equity: pd.Series | None = None,
) -> dict[str, dict[str, float]]:
    split_idx = max(1, min(len(equity) - 1, int(len(equity) * train_ratio)))
    split_date = equity.index[split_idx]

    train_equity = equity.loc[:split_date]
    test_equity = equity.loc[split_date:]
    train_trades = [trade for trade in trades if trade["exit_date"] <= split_date]
    test_trades = [trade for trade in trades if trade["exit_date"] > split_date]

    train_benchmark = benchmark_equity.loc[:split_date] if benchmark_equity is not None else None
    test_benchmark = benchmark_equity.loc[split_date:] if benchmark_equity is not None else None

    return {
        "in_sample": calculate_metrics(train_equity, train_trades, train_benchmark),
        "out_of_sample": calculate_metrics(test_equity, test_trades, test_benchmark),
    }
