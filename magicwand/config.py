"""MAGICWAND — configuration.

Everything tunable lives here and nowhere else, so that when a number
changes we know exactly what changed it.

The single most important field is `live`. It is False and stays False
until a paper strategy has passed its reality check. Read the README.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MarketConfig:
    exchange: str = "coinbase"   # coinbase | kraken | synthetic
    symbol: str = "SOL-USD"      # SOL is affordable at small account sizes
    timeframe: str = "1h"        # 1m 5m 15m 1h 6h 1d
    candles: int = 300           # coinbase allows 300 per request


@dataclass
class CostConfig:
    """Assumptions that kill most strategies. Keep them pessimistic."""
    fee_bps: float = 60.0        # 0.60% per side — Coinbase small-account taker
    slippage_bps: float = 10.0   # 0.10% per side — modest for liquid pairs


@dataclass
class StrategyConfig:
    name: str = "sma_trend"
    fast: int = 12
    slow: int = 48
    atr_period: int = 14
    z_entry: float = 1.5         # mean_revert: entry threshold in std devs
    stop_atr: float = 2.0        # stop distance = stop_atr * ATR
    target_atr: float = 4.0      # take profit distance (0 disables)
    max_hold_bars: int = 72      # time stop: 72h on the 1h timeframe


@dataclass
class RiskConfig:
    """Hard limits. The risk engine enforces these; the strategy cannot."""
    equity: float = 50.0            # paper starting equity, quote currency
    risk_per_trade: float = 0.01    # 1% of equity risked per trade, max
    max_position_pct: float = 0.25  # never more than 25% of equity in one position
    max_daily_loss_pct: float = 0.02  # -2% on the day => stop opening trades
    max_open_positions: int = 1
    min_notional: float = 5.0       # exchange minimum; below this, skip (log why)
    kill_switch: bool = False       # operator panic button


@dataclass
class LedgerConfig:
    path: str = "data/ledger.jsonl"
    state_path: str = "data/state.json"
    claim_horizon_bars: int = 24     # how long a claim has to prove itself


@dataclass
class Config:
    market: MarketConfig = field(default_factory=MarketConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)

    # ---- safety rails (not tunable at runtime) ----
    live: bool = False        # no live execution exists in this build
    seed: int = 7             # deterministic synthetic data / shuffles
    compare_to_hold: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            market=MarketConfig(**d.get("market", {})),
            costs=CostConfig(**d.get("costs", {})),
            strategy=StrategyConfig(**d.get("strategy", {})),
            risk=RiskConfig(**d.get("risk", {})),
            ledger=LedgerConfig(**d.get("ledger", {})),
            live=False,  # never trust a file to turn on live trading
            seed=int(d.get("seed", 7)),
            compare_to_hold=bool(d.get("compare_to_hold", True)),
        )
