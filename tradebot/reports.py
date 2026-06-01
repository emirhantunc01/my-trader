from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import PortfolioBacktestResult
from .pairs import PairBacktestResult
from .risk import split_metrics


KEY_ORDER = [
    "initial_capital",
    "final_capital",
    "total_return_pct",
    "benchmark_return_pct",
    "alpha_vs_benchmark_pct",
    "cagr_pct",
    "volatility_pct",
    "max_drawdown_pct",
    "benchmark_max_drawdown_pct",
    "sharpe",
    "sortino",
    "calmar",
    "recovery_factor",
    "profit_factor",
    "win_rate_pct",
    "trade_count",
    "average_trade_pct",
    "market_exposure_pct",
    "beta_vs_benchmark",
]


def _format_metric(key: str, value: float) -> str:
    if key.endswith("_pct"):
        return f"{value:,.2f}%"
    if key in {"initial_capital", "final_capital"}:
        return f"${value:,.2f}"
    if key == "trade_count":
        return f"{int(value)}"
    if value == float("inf"):
        return "inf"
    return f"{value:,.2f}"


def print_research_disclaimer() -> None:
    print(
        "Note: This is a research backtest, not a performance guarantee. "
        "Results depend on data quality, assumptions, costs, liquidity, and future regimes."
    )


def interpret_metrics(metrics: dict[str, float]) -> list[str]:
    notes: list[str] = []

    alpha = metrics.get("alpha_vs_benchmark_pct")
    benchmark_return = metrics.get("benchmark_return_pct")
    total_return = metrics.get("total_return_pct", 0.0)
    strategy_dd = metrics.get("max_drawdown_pct")
    benchmark_dd = metrics.get("benchmark_max_drawdown_pct")
    sharpe = metrics.get("sharpe", 0.0)
    sortino = metrics.get("sortino", 0.0)
    trade_count = metrics.get("trade_count", 0.0)
    exposure = metrics.get("market_exposure_pct")

    if alpha is not None and benchmark_return is not None:
        if alpha > 0:
            notes.append(
                f"Return beat benchmark by {alpha:.2f} percentage points in this sample."
            )
        else:
            notes.append(
                f"Return lagged benchmark by {abs(alpha):.2f} percentage points in this sample."
            )

    if strategy_dd is not None and benchmark_dd is not None:
        if abs(strategy_dd) < abs(benchmark_dd):
            notes.append(
                f"Drawdown was smaller than benchmark ({strategy_dd:.2f}% vs {benchmark_dd:.2f}%)."
            )
        else:
            notes.append(
                f"Drawdown was not better than benchmark ({strategy_dd:.2f}% vs {benchmark_dd:.2f}%)."
            )

    if sharpe < 0.5:
        notes.append("Sharpe is weak; risk-adjusted performance needs skepticism.")
    elif sharpe < 1.0:
        notes.append("Sharpe is moderate; validate with out-of-sample results.")
    else:
        notes.append("Sharpe is healthy for this sample, but still not proof of future edge.")

    if sortino < sharpe:
        notes.append("Downside volatility is material; inspect losing periods before trusting it.")

    if trade_count < 10:
        notes.append("Very few trades; statistics are fragile.")
    elif trade_count > 150:
        notes.append("High trade count; costs, spread, and overfitting risk matter more.")

    if exposure is not None and exposure < 35:
        notes.append(
            "Low market exposure; compare CAGR and idle cash behavior, not only total return."
        )

    if total_return > 100 and trade_count > 100:
        notes.append("Large return with many trades; run walk-forward before treating it as edge.")

    return notes


def print_interpretation(metrics: dict[str, float], title: str = "Interpretation") -> None:
    notes = interpret_metrics(metrics)
    if not notes:
        return
    print(f"\n{title}:")
    for note in notes:
        print(f"- {note}")


def print_metrics(metrics: dict[str, float], title: str = "Performance") -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    for key in KEY_ORDER:
        if key in metrics:
            print(f"{key:<28}: {_format_metric(key, metrics[key])}")
    print("=" * 72)


def print_portfolio_report(
    result: PortfolioBacktestResult,
    train_ratio: float | None = None,
) -> None:
    print_research_disclaimer()
    print_metrics(result.metrics, "Portfolio backtest")
    print_interpretation(result.metrics)

    if train_ratio is not None:
        splits = split_metrics(result.equity, result.trades, train_ratio, result.benchmark)
        print_metrics(splits["in_sample"], "In-sample segment")
        print_interpretation(splits["in_sample"], "In-sample interpretation")
        print_metrics(splits["out_of_sample"], "Out-of-sample segment")
        print_interpretation(splits["out_of_sample"], "Out-of-sample interpretation")

    if result.trades:
        print("\nLast closed trades:")
        for trade in result.trades[-10:]:
            print(
                f"{trade['symbol']:<8} {trade['leg']:<9} "
                f"{trade['entry_date'].date()} -> {trade['exit_date'].date()} "
                f"return={trade['return_pct']:.2f}% pnl=${trade['pnl']:.2f} "
                f"{trade['reason']}"
            )
    else:
        print("\nNo closed trades.")


def print_pair_report(result: PairBacktestResult) -> None:
    print_research_disclaimer()
    print("\nPair diagnostics:")
    for key, value in result.diagnostics.items():
        if isinstance(value, float):
            print(f"{key:<24}: {value:.2f}")
        else:
            print(f"{key:<24}: {value}")
    if result.diagnostics.get("verdict") == "weak":
        print("Warning: This pair looks statistically weak; treat the backtest as exploratory.")
    elif result.diagnostics.get("verdict") == "borderline":
        print("Warning: This pair is borderline; validate on another period before trusting it.")

    print_metrics(result.metrics, "Pairs trading backtest")
    print_interpretation(result.metrics)
    if result.trades:
        print("\nLast closed pair trades:")
        for trade in result.trades[-10:]:
            print(
                f"{trade['entry_date'].date()} -> {trade['exit_date'].date()} "
                f"return={trade['return_pct']:.2f}% pnl=${trade['pnl']:.2f} "
                f"{trade['reason']}"
            )


def save_portfolio_report(result: PortfolioBacktestResult, output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"strategy": result.equity, "benchmark": result.benchmark}).to_csv(
        report_dir / "equity_curve.csv"
    )
    pd.DataFrame(result.trades).to_csv(report_dir / "trades.csv", index=False)
    pd.DataFrame(result.transactions).to_csv(report_dir / "transactions.csv", index=False)
    pd.DataFrame([result.metrics]).to_csv(report_dir / "metrics.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        ax = result.equity.plot(label="Strategy", figsize=(11, 6))
        result.benchmark.plot(ax=ax, label="Buy & Hold")
        ax.set_title("Equity Curve")
        ax.set_ylabel("Portfolio value")
        ax.legend()
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(report_dir / "equity_curve.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - report generation should not break backtests
        (report_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    return report_dir


def save_pair_report(result: PairBacktestResult, output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    result.equity.to_frame("equity").to_csv(report_dir / "pair_equity_curve.csv")
    pd.DataFrame(result.trades).to_csv(report_dir / "pair_trades.csv", index=False)
    pd.DataFrame([result.metrics]).to_csv(report_dir / "pair_metrics.csv", index=False)
    pd.DataFrame([result.diagnostics]).to_csv(report_dir / "pair_diagnostics.csv", index=False)
    result.spread.to_csv(report_dir / "pair_spread.csv")

    return report_dir


def compare_rows(results: dict[str, PortfolioBacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result.metrics
        rows.append(
            {
                "strategy": name,
                "total_return_pct": metrics.get("total_return_pct", 0.0),
                "benchmark_return_pct": metrics.get("benchmark_return_pct", 0.0),
                "alpha_pct": metrics.get("alpha_vs_benchmark_pct", 0.0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "trade_count": metrics.get("trade_count", 0.0),
                "market_exposure_pct": metrics.get("market_exposure_pct", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "total_return_pct"], ascending=False)


def print_compare_report(results: dict[str, PortfolioBacktestResult]) -> None:
    print_research_disclaimer()
    table = compare_rows(results)
    print("\nStrategy comparison:")
    if table.empty:
        print("No strategy results.")
        return
    display = table.copy()
    for column in display.columns:
        if column != "strategy":
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    print(display.to_string(index=False))

    best_name = str(table.iloc[0]["strategy"])
    print(f"\nBest by Sharpe in this sample: {best_name}")
    print_interpretation(results[best_name].metrics, "Best-sample interpretation")


def save_compare_report(
    results: dict[str, PortfolioBacktestResult],
    output_dir: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"compare_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    compare_rows(results).to_csv(report_dir / "strategy_comparison.csv", index=False)
    for name, result in results.items():
        safe_name = name.replace("-", "_")
        pd.DataFrame({"strategy": result.equity, "benchmark": result.benchmark}).to_csv(
            report_dir / f"{safe_name}_equity_curve.csv"
        )
        pd.DataFrame(result.trades).to_csv(report_dir / f"{safe_name}_trades.csv", index=False)
        pd.DataFrame([result.metrics]).to_csv(report_dir / f"{safe_name}_metrics.csv", index=False)

    return report_dir
