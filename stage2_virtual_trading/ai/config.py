from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class WalkForwardConfig:
    train_months: int = 24
    val_months: int = 6
    test_months: int = 6
    horizon_ticks: int = 60


@dataclass
class SelectorConfig:
    model_type: str = "lightgbm"
    top_k: int = 20
    min_liquidity: float = 1_000_000.0
    risk_lambda: float = 0.5
    horizon_ticks: int = 60
    learning_rate: float = 0.02
    n_estimators: int = 1500
    num_leaves: int = 64
    random_state: int = 42


@dataclass
class FeesConfig:
    buy: float = 0.00015
    sell: float = 0.00015
    tax_sell: float = 0.0005


@dataclass
class SlippageConfig:
    spread_bps: float = 5.0
    base_slip_bps: float = 3.0
    impact_perc: float = 0.05

    def to_kwargs(self) -> Dict[str, float]:
        return {
            "spread_bps": self.spread_bps,
            "base_slip_bps": self.base_slip_bps,
            "impact_perc": self.impact_perc,
        }


@dataclass
class MockTrainingConfig:
    total_timesteps: int = 50_000
    window: int = 60
    seed: int = 7
    algo: str = "ppo"
    num_envs: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ent_coef: float = 0.005
    device: Optional[str] = None


selector_default_config = SelectorConfig()
rl_default_config = MockTrainingConfig()
