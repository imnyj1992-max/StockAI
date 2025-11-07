"""Stage 2 AI training package.

This package implements the hybrid (selector + RL executor) workflow described
in the project specification.  It intentionally keeps concrete dependencies
optional so the module can run in environments without LightGBM, Gymnasium, or
Stable-Baselines installed while still exposing the same API surface.
"""

from .config import (
    FeesConfig,
    MockTrainingConfig,
    SelectorConfig,
    SlippageConfig,
    WalkForwardConfig,
    rl_default_config,
    selector_default_config,
)
from .data_pipeline import DatasetBundle, generate_synthetic_universe, prepare_datasets
from .rl_trainer import train_rl_agent
from .selector import run_selector_stage
from .tick_env import TickTradingEnv, make_vec_envs

__all__ = [
    "DatasetBundle",
    "FeesConfig",
    "MockTrainingConfig",
    "SelectorConfig",
    "SlippageConfig",
    "TickTradingEnv",
    "WalkForwardConfig",
    "generate_synthetic_universe",
    "make_vec_envs",
    "prepare_datasets",
    "rl_default_config",
    "run_selector_stage",
    "selector_default_config",
    "train_rl_agent",
]
