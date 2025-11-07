from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import SelectorConfig, WalkForwardConfig


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    targets: pd.Series
    volatility: pd.Series
    meta: Dict[str, RobustScaler]


def generate_synthetic_universe(
    num_tickers: int = 50,
    num_minutes: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV + indicator panel for quick experiments."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{idx:03d}" for idx in range(num_tickers)]
    minutes = pd.date_range("2022-01-01", periods=num_minutes, freq="T")

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        drift = rng.normal(0, 0.0002)
        shocks = rng.normal(drift, 0.01, size=num_minutes)
        price = 100 * np.exp(np.cumsum(shocks))
        high = price * (1 + rng.normal(0.001, 0.001, size=num_minutes))
        low = price * (1 - rng.normal(0.001, 0.001, size=num_minutes))
        open_ = price * (1 + rng.normal(0, 0.0005, size=num_minutes))
        volume = np.abs(rng.normal(1_000_000, 150_000, size=num_minutes))
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            },
            index=minutes,
        )
        frame["ticker"] = ticker
        frames.append(frame)

    data = pd.concat(frames)
    data.set_index("ticker", append=True, inplace=True)
    data = data.swaplevel(0, 1).sort_index()
    data["returns"] = data.groupby(level=0)["close"].pct_change().fillna(0.0)
    data["log_volume"] = np.log1p(data["volume"])
    data["sma_20"] = data.groupby(level=0)["close"].transform(lambda s: s.rolling(20).mean())
    data["ema_12"] = data.groupby(level=0)["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    data["rsi_14"] = data.groupby(level=0)["returns"].transform(_rsi14)
    data["atr_14"] = data.groupby(level=0).apply(_atr14).droplevel(0)
    data.fillna(method="ffill", inplace=True)
    data.dropna(inplace=True)
    return data


def prepare_datasets(
    data: pd.DataFrame,
    wf: WalkForwardConfig,
) -> DatasetBundle:
    """Create feature/target matrices with walk-forward aware scalers."""
    horizon = wf.horizon_ticks
    grouped = data.groupby(level=0)
    future = grouped["close"].shift(-horizon)
    returns = (future / data["close"]) - 1.0
    volatility = grouped["returns"].transform(lambda s: s.rolling(horizon).std()).fillna(0.0)

    feature_cols = [
        "close",
        "returns",
        "log_volume",
        "sma_20",
        "ema_12",
        "rsi_14",
        "atr_14",
    ]
    features = data[feature_cols].copy()

    log_scaler = RobustScaler()
    indicator_scaler = StandardScaler()

    close_cols = ["close", "log_volume"]
    indicator_cols = [col for col in feature_cols if col not in close_cols]

    features.loc[:, close_cols] = log_scaler.fit_transform(features[close_cols])
    features.loc[:, indicator_cols] = indicator_scaler.fit_transform(features[indicator_cols])

    meta = {
        "log_scaler": log_scaler,
        "indicator_scaler": indicator_scaler,
        "feature_cols": feature_cols,
        "horizon": horizon,
    }
    valid_mask = (~returns.isna()) & (~volatility.isna())
    bundle = DatasetBundle(
        features=features[valid_mask],
        targets=returns[valid_mask],
        volatility=volatility[valid_mask] + 1e-9,
        meta=meta,
    )
    return bundle


def walk_forward_slices(
    bundle: DatasetBundle,
    wf: WalkForwardConfig,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute walk-forward train/val/test integer slices."""
    dates = bundle.features.index.get_level_values(1).unique().sort_values()
    step = wf.test_months
    slices = []
    total_months = math.floor(len(dates) / 30)
    if total_months < wf.train_months + wf.val_months + wf.test_months:
        # fallback single split
        idx = np.arange(len(bundle.features))
        slices.append((idx[:-200], idx[-200:-100], idx[-100:]))
        return slices

    for start in range(0, total_months - (wf.train_months + wf.val_months + wf.test_months), step):
        train_end = start + wf.train_months
        val_end = train_end + wf.val_months
        test_end = val_end + wf.test_months
        mask_train = (dates >= dates[start]) & (dates < dates[train_end])
        mask_val = (dates >= dates[train_end]) & (dates < dates[val_end])
        mask_test = (dates >= dates[val_end]) & (dates < dates[test_end])
        idx_train = np.where(bundle.features.index.get_level_values(1).isin(dates[mask_train]))[0]
        idx_val = np.where(bundle.features.index.get_level_values(1).isin(dates[mask_val]))[0]
        idx_test = np.where(bundle.features.index.get_level_values(1).isin(dates[mask_test]))[0]
        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            continue
        slices.append((idx_train, idx_val, idx_test))
    if not slices:
        idx = np.arange(len(bundle.features))
        slices.append((idx[:-200], idx[-200:-100], idx[-100:]))
    return slices


def _rsi14(series: pd.Series) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr14(group: pd.DataFrame) -> pd.Series:
    high_low = group["high"] - group["low"]
    high_close = (group["high"] - group["close"].shift()).abs()
    low_close = (group["low"] - group["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return atr
