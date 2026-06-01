from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tradebot.backtest import run_portfolio_backtest
from tradebot.config import BacktestConfig
from tradebot.data import download_universe
from tradebot.pairs import run_pairs_backtest
from tradebot.reports import (
    print_compare_report,
    print_pair_report,
    print_portfolio_report,
    save_compare_report,
    save_pair_report,
    save_portfolio_report,
)
from tradebot.strategies import strategy_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-grade algorithmic trading backtester.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL"],
        help="Symbols to test. Example: --symbols AAPL MSFT NVDA SPY",
    )
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 2y, 5y, 10y")
    parser.add_argument("--interval", default="1d", help="yfinance interval, e.g. 1d, 1h")
    parser.add_argument(
        "--strategy",
        default="core-satellite",
        choices=["core-satellite", "trend", "mean-reversion", "hybrid", "pairs"],
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run core-satellite, trend, mean-reversion, and hybrid on the same data.",
    )
    parser.add_argument(
        "--compare-strategies",
        nargs="+",
        default=["core-satellite", "trend", "mean-reversion", "hybrid"],
        choices=["core-satellite", "trend", "mean-reversion", "hybrid"],
        help="Strategies to include when --compare is used.",
    )
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--core-position-pct", type=float, default=0.90)
    parser.add_argument("--target-position-pct", type=float, default=0.95)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.50)
    parser.add_argument("--atr-stop-multiplier", type=float, default=8.0)
    parser.add_argument("--adx-threshold", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--save-reports", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("SYMBOL_A", "SYMBOL_B"),
        help="Run pairs trading on exactly two symbols.",
    )
    parser.add_argument("--pair-lookback", type=int, default=60)
    parser.add_argument("--pair-entry-z", type=float, default=2.0)
    parser.add_argument("--pair-exit-z", type=float, default=0.5)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BacktestConfig:
    symbols = tuple(args.pair if args.strategy == "pairs" and args.pair else args.symbols)
    return BacktestConfig(
        symbols=tuple(symbol.upper() for symbol in symbols),
        period=args.period,
        interval=args.interval,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        strategy=args.strategy,
        core_position_pct=args.core_position_pct,
        target_position_pct=args.target_position_pct,
        risk_per_trade_pct=args.risk_per_trade_pct,
        atr_stop_multiplier=args.atr_stop_multiplier,
        adx_threshold=args.adx_threshold,
        train_ratio=args.train_ratio,
        save_reports=args.save_reports,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    data = download_universe(
        config.normalized_symbols,
        period=config.period,
        interval=config.interval,
    )

    if args.compare:
        results = {}
        for strategy_name in args.compare_strategies:
            strategy_config = replace(config, strategy=strategy_name)
            strategy, core_position_pct = strategy_factory(strategy_config)
            results[strategy_name] = run_portfolio_backtest(
                data=data,
                strategy=strategy,
                config=strategy_config,
                core_position_pct=core_position_pct,
            )
        print_compare_report(results)
        if config.save_reports:
            report_dir = save_compare_report(results, config.output_dir)
            print(f"\nSaved reports to: {report_dir}")
        return

    if config.strategy == "pairs":
        if len(config.normalized_symbols) != 2:
            raise ValueError("Pairs strategy requires exactly two symbols.")
        result = run_pairs_backtest(
            config.normalized_symbols[0],
            config.normalized_symbols[1],
            data,
            config,
            lookback=args.pair_lookback,
            entry_z=args.pair_entry_z,
            exit_z=args.pair_exit_z,
        )
        print_pair_report(result)
        if config.save_reports:
            report_dir = save_pair_report(result, config.output_dir)
            print(f"\nSaved reports to: {report_dir}")
        return

    strategy, core_position_pct = strategy_factory(config)
    result = run_portfolio_backtest(
        data=data,
        strategy=strategy,
        config=config,
        core_position_pct=core_position_pct,
    )
    print_portfolio_report(
        result,
        train_ratio=config.train_ratio if args.walk_forward else None,
    )

    if config.save_reports:
        report_dir = save_portfolio_report(result, config.output_dir)
        print(f"\nSaved reports to: {report_dir}")


if __name__ == "__main__":
    main()
