from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    symbols: tuple[str, ...]
    period: str = "5y"
    interval: str = "1d"
    initial_capital: float = 10_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    strategy: str = "core-satellite"
    core_position_pct: float = 0.90
    target_position_pct: float = 0.95
    risk_per_trade_pct: float = 0.50
    atr_stop_multiplier: float = 8.0
    adx_threshold: float = 15.0
    train_ratio: float = 0.70
    save_reports: bool = False
    output_dir: Path = Path("outputs")

    @property
    def normalized_symbols(self) -> tuple[str, ...]:
        return tuple(symbol.upper() for symbol in self.symbols)
