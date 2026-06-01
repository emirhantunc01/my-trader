from __future__ import annotations

import argparse
import getpass
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tradebot.backtest import run_portfolio_backtest
from tradebot.config import BacktestConfig
from tradebot.data import download_universe
from tradebot.mt5_adapter import account_snapshot, fetch_universe_rates, latest_bid_ask, mt5_session
from tradebot.mt5_trader import evaluate_and_execute_demo
from tradebot.pairs import run_pairs_backtest
from tradebot.paper import evaluate_symbol_once, load_state, paper_summary, save_state
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
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run one paper-trading evaluation cycle. No live/demo orders are sent.",
    )
    parser.add_argument(
        "--paper-source",
        choices=["mt5", "yfinance"],
        default="mt5",
        help="Price source for --paper. Use mt5 for MetaTrader 5 live terminal data.",
    )
    parser.add_argument("--paper-state", default="paper_state.json")
    parser.add_argument("--paper-balance", type=float, default=10_000.0)
    parser.add_argument("--paper-reset", action="store_true")
    parser.add_argument("--paper-bars", type=int, default=350)
    parser.add_argument("--mt5-timeframe", default="M15")
    parser.add_argument("--mt5-terminal-path")
    parser.add_argument("--mt5-login", type=int)
    parser.add_argument("--mt5-password")
    parser.add_argument("--mt5-server")
    parser.add_argument(
        "--execute-demo",
        action="store_true",
        help="Send real orders to an MT5 demo account when strategy signals fire.",
    )
    parser.add_argument(
        "--confirm-demo-orders",
        action="store_true",
        help="Required safety confirmation for --execute-demo.",
    )
    parser.add_argument("--demo-volume", type=float, default=0.01)
    parser.add_argument("--demo-deviation", type=int, default=20)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep demo execution running until Ctrl+C instead of running one cycle.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait between loop cycles. Use 1 for once-per-second checks.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Stop after this many cycles. Useful for testing loop mode.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BacktestConfig:
    symbols = tuple(args.pair if args.strategy == "pairs" and args.pair else args.symbols)
    return BacktestConfig(
        symbols=tuple(symbol.strip() for symbol in symbols),
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


def run_paper(args: argparse.Namespace, config: BacktestConfig) -> None:
    if config.strategy == "pairs":
        raise ValueError("Paper mode supports directional strategies, not pairs.")

    paper_config = replace(
        config,
        initial_capital=args.paper_balance,
        strategy="trend" if config.strategy == "core-satellite" else config.strategy,
        core_position_pct=0.0,
    )
    strategy, _ = strategy_factory(paper_config)
    state_path = Path(args.paper_state)
    state = load_state(state_path, args.paper_balance, reset=args.paper_reset)

    if args.paper_source == "mt5":
        mt5_password = args.mt5_password or os.environ.get("MT5_PASSWORD")
        if args.mt5_login and args.mt5_server and not mt5_password:
            mt5_password = getpass.getpass("MT5 password: ")

        with mt5_session(
            terminal_path=args.mt5_terminal_path,
            login=args.mt5_login,
            password=mt5_password,
            server=args.mt5_server,
        ) as mt5:
            snapshot = account_snapshot(mt5)
            print("MT5 account snapshot:")
            for key, value in snapshot.items():
                print(f"{key:<12}: {value}")

            mt5_symbols = tuple(config.symbols)
            data = fetch_universe_rates(
                mt5,
                mt5_symbols,
                timeframe=args.mt5_timeframe,
                bars=args.paper_bars,
            )
            bid_ask = {
                symbol: latest_bid_ask(mt5, symbol)
                for symbol in mt5_symbols
            }
    else:
        data = download_universe(
            config.normalized_symbols,
            period=config.period,
            interval=config.interval,
        )
        bid_ask = {}
        for symbol, df in data.items():
            last_close = float(df["Close"].iloc[-1])
            bid_ask[symbol] = (
                last_close * (1 - config.slippage),
                last_close * (1 + config.slippage),
            )

    decisions = []
    market_prices = {}
    for symbol, df in data.items():
        bid, ask = bid_ask[symbol]
        market_prices[symbol] = bid
        decision = evaluate_symbol_once(
            symbol=symbol,
            raw_df=df,
            bid=bid,
            ask=ask,
            state=state,
            strategy=strategy,
            config=paper_config,
        )
        decisions.append(decision)

    save_state(state_path, state)
    summary = paper_summary(state, market_prices)

    print("\nPaper decisions:")
    for decision in decisions:
        print(
            f"{decision.symbol:<12} {decision.action:<5} "
            f"qty={decision.quantity:.6f} price={decision.price:.5f} "
            f"{decision.reason}"
        )

    print("\nPaper portfolio:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<16}: {value:,.2f}")
        else:
            print(f"{key:<16}: {value}")
    print(f"\nState saved to: {state_path}")
    print("Safety: no order_send call is used in paper mode.")


def run_execute_demo(args: argparse.Namespace, config: BacktestConfig) -> None:
    if not args.confirm_demo_orders:
        raise RuntimeError(
            "--execute-demo requires --confirm-demo-orders. "
            "This sends real orders to the connected MT5 demo account."
        )
    if config.strategy == "pairs":
        raise ValueError("Demo execution supports directional strategies, not pairs.")

    exec_config = replace(
        config,
        strategy="trend" if config.strategy == "core-satellite" else config.strategy,
        core_position_pct=0.0,
    )
    strategy, _ = strategy_factory(exec_config)
    mt5_password = args.mt5_password or os.environ.get("MT5_PASSWORD")
    if args.mt5_login and args.mt5_server and not mt5_password:
        mt5_password = getpass.getpass("MT5 password: ")

    with mt5_session(
        terminal_path=args.mt5_terminal_path,
        login=args.mt5_login,
        password=mt5_password,
        server=args.mt5_server,
    ) as mt5:
        snapshot = account_snapshot(mt5)
        print("MT5 account snapshot:")
        for key, value in snapshot.items():
            print(f"{key:<12}: {value}")

        cycle = 0
        sleep_seconds = max(1.0, float(args.sleep_seconds))
        if args.loop:
            print(
                "\nLoop mode enabled. "
                f"Checking {len(config.symbols)} symbol(s) every {sleep_seconds:g}s. "
                "Press Ctrl+C to stop."
            )

        try:
            while True:
                cycle += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\nDemo execution decisions | cycle={cycle} | {timestamp}")

                try:
                    data = fetch_universe_rates(
                        mt5,
                        tuple(config.symbols),
                        timeframe=args.mt5_timeframe,
                        bars=args.paper_bars,
                    )

                    for symbol, df in data.items():
                        decision = evaluate_and_execute_demo(
                            mt5=mt5,
                            symbol=symbol,
                            raw_df=df,
                            strategy=strategy,
                            config=exec_config,
                            volume=args.demo_volume,
                            deviation=args.demo_deviation,
                        )
                        print(
                            f"{decision.symbol:<12} {decision.action:<12} "
                            f"volume={decision.volume:.4f} price={decision.price:.5f} "
                            f"sl={decision.sl:.5f} retcode={decision.retcode} {decision.reason}"
                        )
                except Exception as exc:
                    print(f"Cycle {cycle} failed: {exc}")
                    if not args.loop:
                        raise

                if not args.loop:
                    break
                if args.max_cycles is not None and cycle >= args.max_cycles:
                    break
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\nStopped by user. MT5 session closed.")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.execute_demo:
        run_execute_demo(args, config)
        return

    if args.paper:
        run_paper(args, config)
        return

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
