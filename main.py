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
from tradebot.mt5_adapter import (
    account_snapshot,
    fetch_universe_rates,
    fetch_universe_timeframe_rates,
    latest_bid_ask,
    mt5_session,
)
from tradebot.mt5_trader import evaluate_and_execute_demo, load_demo_state, save_demo_state
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
        choices=["core-satellite", "trend", "fast-trend", "scalp", "mean-reversion", "hybrid", "pairs"],
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
        choices=["core-satellite", "trend", "fast-trend", "scalp", "mean-reversion", "hybrid"],
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
    parser.add_argument("--paper-bars", type=int, default=500)
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
    parser.add_argument("--demo-stop-atr", type=float, default=1.0)
    parser.add_argument("--demo-take-profit-atr", type=float, default=0.5)
    parser.add_argument("--demo-max-hold-minutes", type=float, default=15.0)
    parser.add_argument("--demo-analysis-timeframe", help="Primary demo timeframe. Scalp defaults to M1.")
    parser.add_argument("--demo-confirm-timeframe", default="M5")
    parser.add_argument("--demo-signal-timeframe", default="M15")
    parser.add_argument("--demo-max-positions-per-direction", type=int, default=3)
    parser.add_argument("--demo-pyramid-step-pct", type=float, default=0.005)
    parser.add_argument("--demo-pyramid-add-volume-pct", type=float, default=0.50)
    parser.add_argument("--demo-pyramid-stop-atr", type=float, default=1.5)
    parser.add_argument("--demo-partial-first-profit-pct", type=float, default=0.003)
    parser.add_argument("--demo-partial-second-profit-pct", type=float, default=0.006)
    parser.add_argument("--demo-partial-first-volume-pct", type=float, default=0.30)
    parser.add_argument("--demo-partial-second-volume-pct", type=float, default=0.50)
    parser.add_argument("--demo-trailing-stop-atr", type=float, default=1.2)
    parser.add_argument("--demo-signal-cooldown-bars", type=int, default=2)
    parser.add_argument("--demo-close-all-profit-pct", type=float, default=0.01)
    parser.add_argument("--demo-allow-short", action="store_true")
    parser.add_argument("--demo-state", default="demo_state.json")
    parser.add_argument("--demo-reset", action="store_true")
    parser.add_argument(
        "--demo-dry-run",
        action="store_true",
        help="Evaluate demo signals without sending or closing MT5 orders.",
    )
    parser.add_argument(
        "--debug-signals",
        action="store_true",
        help="Print indicator values and signal booleans for each evaluated symbol.",
    )
    parser.add_argument(
        "--demo-trade-current-signal",
        action="store_true",
        help="Allow entering on the already-active signal seen when the bot starts.",
    )
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
    demo_state_path = Path(args.demo_state)
    demo_state = load_demo_state(demo_state_path, reset=args.demo_reset)
    mt5_password = args.mt5_password or os.environ.get("MT5_PASSWORD")
    if args.mt5_login and args.mt5_server and not mt5_password:
        mt5_password = getpass.getpass("MT5 password: ")
    is_scalp = exec_config.strategy == "scalp"
    analysis_timeframe = args.demo_analysis_timeframe or ("M1" if is_scalp else args.mt5_timeframe)
    history_bars = max(int(args.paper_bars), 500) if analysis_timeframe.upper() == "M1" else int(args.paper_bars)

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
        if args.demo_dry_run:
            print("Mode        : DRY_RUN (signals only, no MT5 orders)")
        else:
            print("Mode        : LIVE_DEMO_ORDERS (bot sends/closes demo orders)")
        print(f"Timeframes  : primary={analysis_timeframe}", end="")
        if is_scalp:
            print(f" confirm={args.demo_confirm_timeframe} signal={args.demo_signal_timeframe}", end="")
        print(f" bars={history_bars}")

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
                    if is_scalp:
                        data = fetch_universe_timeframe_rates(
                            mt5,
                            tuple(config.symbols),
                            timeframes={
                                "primary": analysis_timeframe,
                                "confirm": args.demo_confirm_timeframe,
                                "signal": args.demo_signal_timeframe,
                            },
                            bars={
                                "primary": history_bars,
                                "confirm": max(200, history_bars),
                                "signal": max(200, history_bars),
                            },
                        )
                    else:
                        data = fetch_universe_rates(
                            mt5,
                            tuple(config.symbols),
                            timeframe=analysis_timeframe,
                            bars=history_bars,
                        )

                    for symbol, frames in data.items():
                        if is_scalp:
                            df = frames["primary"]
                            confirm_df = frames["confirm"]
                            signal_df = frames["signal"]
                        else:
                            df = frames
                            confirm_df = None
                            signal_df = None
                        decision = evaluate_and_execute_demo(
                            mt5=mt5,
                            symbol=symbol,
                            raw_df=df,
                            strategy=strategy,
                            config=exec_config,
                            volume=args.demo_volume,
                            deviation=args.demo_deviation,
                            stop_atr=args.demo_stop_atr,
                            take_profit_atr=args.demo_take_profit_atr,
                            max_hold_minutes=args.demo_max_hold_minutes,
                            allow_short=args.demo_allow_short,
                            state=None if args.demo_dry_run else demo_state,
                            trade_current_signal=args.demo_trade_current_signal,
                            dry_run=args.demo_dry_run,
                            debug_signals=args.debug_signals,
                            confirm_raw_df=confirm_df,
                            signal_raw_df=signal_df,
                            max_positions_per_direction=args.demo_max_positions_per_direction,
                            pyramid_step_pct=args.demo_pyramid_step_pct,
                            pyramid_add_volume_pct=args.demo_pyramid_add_volume_pct,
                            pyramid_stop_atr=args.demo_pyramid_stop_atr,
                            partial_first_profit_pct=args.demo_partial_first_profit_pct,
                            partial_second_profit_pct=args.demo_partial_second_profit_pct,
                            partial_first_volume_pct=args.demo_partial_first_volume_pct,
                            partial_second_volume_pct=args.demo_partial_second_volume_pct,
                            trailing_stop_atr=args.demo_trailing_stop_atr,
                            signal_cooldown_bars=args.demo_signal_cooldown_bars,
                            close_all_profit_pct=args.demo_close_all_profit_pct,
                        )
                        position_text = ""
                        if decision.position_volume > 0:
                            position_text = (
                                f" pos={decision.position_side}:{decision.position_volume:.4f}"
                                f" entry={decision.position_entry:.5f}"
                                f" pnl={decision.position_profit:.2f}"
                            )
                        print(
                            f"{decision.symbol:<12} {decision.action:<12} "
                            f"signal={decision.signal:<4} "
                            f"order_volume={decision.volume:.4f} price={decision.price:.5f} "
                            f"sl={decision.sl:.5f} tp={decision.tp:.5f} retcode={decision.retcode}"
                            f"{position_text} {decision.reason}"
                        )
                    if not args.demo_dry_run:
                        save_demo_state(demo_state_path, demo_state)
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
