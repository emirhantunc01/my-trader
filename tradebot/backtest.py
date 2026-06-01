from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import pandas as pd

from .config import BacktestConfig
from .indicators import add_common_indicators
from .risk import calculate_metrics
from .strategies import Strategy


@dataclass
class AssetBacktestResult:
    symbol: str
    equity: pd.Series
    benchmark: pd.Series
    trades: list[dict]
    transactions: list[dict]
    invested_days: int


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    trades: list[dict]
    transactions: list[dict]
    metrics: dict[str, float]
    asset_results: dict[str, AssetBacktestResult]


def _execution_price(price: float, side: str, slippage: float) -> float:
    if side == "buy":
        return price * (1 + slippage)
    if side == "sell":
        return price * (1 - slippage)
    raise ValueError(f"Unknown execution side: {side}")


def _close_long_position(
    symbol: str,
    cash: float,
    shares: float,
    exit_price: float,
    entry_date,
    entry_price: float,
    entry_commission: float,
    exit_date,
    reason: str,
    leg: str,
    config: BacktestConfig,
    trades: list[dict],
    transactions: list[dict],
) -> tuple[float, float]:
    sell_value = shares * exit_price
    sell_commission = sell_value * config.commission
    cash += sell_value - sell_commission

    buy_value = shares * entry_price
    pnl = sell_value - sell_commission - buy_value - entry_commission
    return_pct = pnl / (buy_value + entry_commission) * 100

    trades.append(
        {
            "symbol": symbol,
            "leg": leg,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "return_pct": return_pct,
            "reason": reason,
        }
    )
    transactions.append(
        {
            "symbol": symbol,
            "date": exit_date,
            "type": f"{leg} SELL",
            "price": exit_price,
            "shares": shares,
            "reason": reason,
        }
    )

    return cash, 0.0


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


def buy_hold_equity(
    symbol: str,
    df: pd.DataFrame,
    initial_capital: float,
    config: BacktestConfig,
) -> pd.Series:
    if len(df) < 2:
        return pd.Series(dtype=float, name=f"{symbol}_benchmark")

    cash = initial_capital
    entry_price = _execution_price(float(df["Open"].iloc[1]), "buy", config.slippage)
    shares = cash / (entry_price * (1 + config.commission))
    buy_value = shares * entry_price
    cash -= buy_value + buy_value * config.commission

    values = [(df.index[0], initial_capital)]
    for i in range(1, len(df)):
        values.append((df.index[i], cash + shares * float(df["Close"].iloc[i])))

    final_price = _execution_price(float(df["Close"].iloc[-1]), "sell", config.slippage)
    final_value = cash + shares * final_price * (1 - config.commission)
    values[-1] = (df.index[-1], final_value)

    return pd.Series(
        [value for _, value in values],
        index=[date for date, _ in values],
        name=f"{symbol}_benchmark",
    )


def run_asset_backtest(
    symbol: str,
    raw_df: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    initial_capital: float,
    core_position_pct: float,
) -> AssetBacktestResult:
    df = add_common_indicators(raw_df)
    df = strategy.generate_signals(df)

    cash = initial_capital
    core_shares = 0.0
    core_entry_price = 0.0
    core_entry_commission = 0.0
    core_entry_date = None
    tactical_shares = 0.0
    tactical_entry_price = 0.0
    tactical_entry_commission = 0.0
    tactical_entry_date = None
    trailing_stop = 0.0
    trades: list[dict] = []
    transactions: list[dict] = []
    equity_curve = [(df.index[0], initial_capital)]
    invested_days = 0

    if len(df) > 1 and core_position_pct > 0:
        core_entry_date = df.index[1]
        core_entry_price = _execution_price(float(df["Open"].iloc[1]), "buy", config.slippage)
        core_shares = (cash * core_position_pct) / (
            core_entry_price * (1 + config.commission)
        )
        core_buy_value = core_shares * core_entry_price
        core_entry_commission = core_buy_value * config.commission
        cash -= core_buy_value + core_entry_commission
        transactions.append(
            {
                "symbol": symbol,
                "date": core_entry_date,
                "type": "CORE BUY",
                "price": core_entry_price,
                "shares": core_shares,
                "reason": "CORE_LONG",
            }
        )

    for i in range(1, len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]
        execution_date = df.index[i + 1]
        current_equity = cash + (core_shares + tactical_shares) * float(today["Close"])
        exited = False

        if tactical_shares > 0:
            trailing_stop = max(
                trailing_stop,
                float(today["Close"] - config.atr_stop_multiplier * today["atr"]),
            )
            stop_hit = float(tomorrow["Low"]) <= trailing_stop

            if stop_hit:
                raw_exit = min(float(tomorrow["Open"]), trailing_stop)
                exit_price = _execution_price(raw_exit, "sell", config.slippage)
                cash, tactical_shares = _close_long_position(
                    symbol,
                    cash,
                    tactical_shares,
                    exit_price,
                    tactical_entry_date,
                    tactical_entry_price,
                    tactical_entry_commission,
                    execution_date,
                    "TACTICAL_ATR_STOP",
                    "TACTICAL",
                    config,
                    trades,
                    transactions,
                )
                exited = True
            elif bool(today["sell_signal"]):
                exit_price = _execution_price(float(tomorrow["Open"]), "sell", config.slippage)
                cash, tactical_shares = _close_long_position(
                    symbol,
                    cash,
                    tactical_shares,
                    exit_price,
                    tactical_entry_date,
                    tactical_entry_price,
                    tactical_entry_commission,
                    execution_date,
                    "TACTICAL_SELL_SIGNAL",
                    "TACTICAL",
                    config,
                    trades,
                    transactions,
                )
                exited = True

        if tactical_shares == 0 and not exited and bool(today["buy_signal"]):
            tactical_entry_price = _execution_price(
                float(tomorrow["Open"]),
                "buy",
                config.slippage,
            )
            tactical_shares = _position_size(
                cash,
                current_equity,
                tactical_entry_price,
                float(today["atr"]),
                config,
            )

            if tactical_shares > 0:
                buy_value = tactical_shares * tactical_entry_price
                tactical_entry_commission = buy_value * config.commission
                cash -= buy_value + tactical_entry_commission
                tactical_entry_date = execution_date
                trailing_stop = tactical_entry_price - config.atr_stop_multiplier * float(
                    today["atr"]
                )
                transactions.append(
                    {
                        "symbol": symbol,
                        "date": tactical_entry_date,
                        "type": "TACTICAL BUY",
                        "price": tactical_entry_price,
                        "shares": tactical_shares,
                        "reason": "BUY_SIGNAL",
                    }
                )

        close_equity = cash + (core_shares + tactical_shares) * float(tomorrow["Close"])
        equity_curve.append((execution_date, close_equity))
        if core_shares > 0 or tactical_shares > 0:
            invested_days += 1

    final_date = df.index[-1]
    final_price = _execution_price(float(df["Close"].iloc[-1]), "sell", config.slippage)

    if tactical_shares > 0:
        cash, tactical_shares = _close_long_position(
            symbol,
            cash,
            tactical_shares,
            final_price,
            tactical_entry_date,
            tactical_entry_price,
            tactical_entry_commission,
            final_date,
            "TACTICAL_FINAL_CLOSE",
            "TACTICAL",
            config,
            trades,
            transactions,
        )

    if core_shares > 0:
        cash, core_shares = _close_long_position(
            symbol,
            cash,
            core_shares,
            final_price,
            core_entry_date,
            core_entry_price,
            core_entry_commission,
            final_date,
            "CORE_FINAL_CLOSE",
            "CORE",
            config,
            trades,
            transactions,
        )
        equity_curve[-1] = (final_date, cash)

    equity = pd.Series(
        [value for _, value in equity_curve],
        index=[date for date, _ in equity_curve],
        name=symbol,
    )
    benchmark = buy_hold_equity(symbol, df, initial_capital, config)

    return AssetBacktestResult(
        symbol=symbol,
        equity=equity,
        benchmark=benchmark,
        trades=trades,
        transactions=transactions,
        invested_days=invested_days,
    )


def run_portfolio_backtest(
    data: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    core_position_pct: float,
) -> PortfolioBacktestResult:
    symbols = tuple(data)
    per_symbol_capital = config.initial_capital / len(symbols)
    asset_results: dict[str, AssetBacktestResult] = {}
    all_trades: list[dict] = []
    all_transactions: list[dict] = []

    for symbol, df in data.items():
        asset_config = replace(config, initial_capital=per_symbol_capital)
        result = run_asset_backtest(
            symbol=symbol,
            raw_df=df,
            strategy=strategy,
            config=asset_config,
            initial_capital=per_symbol_capital,
            core_position_pct=core_position_pct,
        )
        asset_results[symbol] = result
        all_trades.extend(result.trades)
        all_transactions.extend(result.transactions)

    equity_df = pd.concat([result.equity for result in asset_results.values()], axis=1)
    equity_df = equity_df.sort_index().ffill().bfill()
    portfolio_equity = equity_df.sum(axis=1)
    portfolio_equity.name = "strategy_equity"

    benchmark_df = pd.concat(
        [result.benchmark for result in asset_results.values()],
        axis=1,
    )
    benchmark_df = benchmark_df.sort_index().ffill().bfill()
    benchmark_equity = benchmark_df.sum(axis=1)
    benchmark_equity.name = "benchmark_equity"

    invested_days = sum(result.invested_days for result in asset_results.values())
    total_days = max(sum(len(result.equity) - 1 for result in asset_results.values()), 1)
    exposure = invested_days / total_days * 100

    metrics = calculate_metrics(
        portfolio_equity,
        all_trades,
        benchmark_equity=benchmark_equity,
        market_exposure=exposure,
    )

    return PortfolioBacktestResult(
        equity=portfolio_equity,
        benchmark=benchmark_equity,
        trades=all_trades,
        transactions=all_transactions,
        metrics=metrics,
        asset_results=asset_results,
    )
