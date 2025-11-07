from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

try:  # pragma: no cover - support direct execution
    if __package__ is None or __package__ == "":
        import sys

        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from ai import (
            DatasetBundle,
            FeesConfig,
            MockTrainingConfig,
            SelectorConfig,
            SlippageConfig,
            TickTradingEnv,
            WalkForwardConfig,
            generate_synthetic_universe,
            make_vec_envs,
            prepare_datasets,
            rl_default_config,
            run_selector_stage,
            selector_default_config,
            train_rl_agent,
        )
    else:
        from .ai import (
            DatasetBundle,
            FeesConfig,
            MockTrainingConfig,
            SelectorConfig,
            SlippageConfig,
            TickTradingEnv,
            WalkForwardConfig,
            generate_synthetic_universe,
            make_vec_envs,
            prepare_datasets,
            rl_default_config,
            run_selector_stage,
            selector_default_config,
            train_rl_agent,
        )
except ImportError as exc:  # pragma: no cover
    raise

log = logging.getLogger("stage2.ai")


def _split_by_selection(bundle: DatasetBundle, selected) -> List[np.ndarray]:
    arrays = []
    prices = []
    for ticker in selected:
        mask = bundle.features.index.get_level_values(0) == ticker
        arrays.append(bundle.features.loc[ticker].values.astype(np.float32))
        prices.append(
            bundle.features.loc[ticker]["close"].values.astype(np.float32)
        )
    return arrays, prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 mock AI training pipeline.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (default).")
    parser.add_argument("--ticks", type=int, default=60, help="Prediction horizon and RL window.")
    parser.add_argument("--topk", type=int, default=20, help="Number of tickers passed to RL.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total RL timesteps.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.synthetic or True:
        data = generate_synthetic_universe(num_tickers=40, num_minutes=5_000)
    else:
        raise NotImplementedError("Custom data loading not wired yet.")

    wf_cfg = WalkForwardConfig(horizon_ticks=args.ticks)
    selector_cfg = SelectorConfig(top_k=args.topk, horizon_ticks=args.ticks)
    bundle = prepare_datasets(data, wf_cfg)
    selector_result = run_selector_stage(bundle, selector_cfg, wf_cfg)

    log.info("Selector metrics: %s", selector_result.metrics)
    arrays, prices = _split_by_selection(bundle, selector_result.selected)
    fees = {"buy": 0.00015, "sell": 0.00015, "tax_sell": 0.0005}
    envs = make_vec_envs(arrays, prices, window=args.ticks, fees=fees, slippage_bps=5.0)
    rl_cfg = MockTrainingConfig(total_timesteps=args.timesteps, window=args.ticks)
    model = train_rl_agent(envs, rl_cfg)
    log.info("RL training done. Model: %s", model.__class__.__name__)


if __name__ == "__main__":
    main()
