from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import SelectorConfig, WalkForwardConfig
from .data_pipeline import DatasetBundle, walk_forward_slices

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class SelectorResult:
    model: object
    predictions: pd.Series
    scores: pd.Series
    metrics: Dict[str, float]
    selected: pd.Index


def run_selector_stage(
    bundle: DatasetBundle,
    cfg: SelectorConfig,
    wf_cfg: WalkForwardConfig,
) -> SelectorResult:
    """Train the stage-A selector and pick top-K tickers for RL stage."""
    slices = walk_forward_slices(bundle, wf_cfg)
    best_model = None
    best_rmse = float("inf")

    X = bundle.features.values
    y = bundle.targets.values

    for idx_train, idx_val, _ in slices:
        model = _create_model(cfg)
        model.fit(X[idx_train], y[idx_train])
        preds = model.predict(X[idx_val])
        rmse = mean_squared_error(y[idx_val], preds, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    if best_model is None:
        # fallback simple train/test split
        idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.2, random_state=cfg.random_state)
        best_model = _create_model(cfg)
        best_model.fit(X[idx_train], y[idx_train])
        preds = best_model.predict(X[idx_val])
        best_rmse = mean_squared_error(y[idx_val], preds, squared=False)

    full_preds = best_model.predict(X)
    scores = _risk_adjust(full_preds, bundle.volatility.values, cfg.risk_lambda)
    scores_series = pd.Series(scores, index=bundle.features.index)

    filtered = scores_series[bundle.features["log_volume"] >= np.log1p(cfg.min_liquidity / 1_000_000)]
    top = filtered.groupby(level=0).tail(1).sort_values(ascending=False)
    selected = top.head(cfg.top_k).index.get_level_values(0)

    metrics = {"rmse": float(best_rmse), "num_selected": len(selected)}
    return SelectorResult(
        model=best_model,
        predictions=pd.Series(full_preds, index=bundle.features.index),
        scores=scores_series,
        metrics=metrics,
        selected=selected,
    )


def _create_model(cfg: SelectorConfig):
    if _HAS_LGB and cfg.model_type.lower().startswith("lightgbm"):
        return lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            random_state=cfg.random_state,
        )
    return GradientBoostingRegressor(
        n_estimators=min(500, cfg.n_estimators),
        learning_rate=cfg.learning_rate,
        max_depth=5,
        random_state=cfg.random_state,
    )


def _risk_adjust(preds: np.ndarray, vol: np.ndarray, lam: float) -> np.ndarray:
    return preds - lam * np.abs(vol)
