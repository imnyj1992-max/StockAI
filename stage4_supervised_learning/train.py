from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class TrainingResult:
    symbol: str
    horizon: int
    current_price: float
    predicted_price: float
    r2: float
    rmse: float
    directional_accuracy: float


def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols:
        df["timestamp"] = pd.to_datetime(df[cols["timestamp"]])
    elif "date" in cols:
        df["timestamp"] = pd.to_datetime(df[cols["date"]])
    else:
        raise ValueError("CSV must contain a 'timestamp' or 'date' column")
    if "close" not in cols:
        raise ValueError("CSV must contain a 'close' column")
    df = df.rename(columns={cols["close"]: "close"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" in cols:
        df = df.rename(columns={cols["volume"]: "volume"})
    else:
        df["volume"] = np.nan
    return df


def generate_synthetic_data(symbol: str, periods: int = 2000, freq: str = "1min") -> pd.DataFrame:
    """Create a synthetic random-walk dataset when no CSV is provided."""
    end_time = datetime.now().replace(second=0, microsecond=0)
    timestamps = pd.date_range(end=end_time, periods=periods, freq=freq)
    rng = np.random.default_rng(42)
    drift = 0.0003
    shock = rng.normal(drift, 0.01, size=periods)
    close = 100 * np.exp(np.cumsum(shock))
    volume = rng.integers(1000, 10000, size=periods)
    df = pd.DataFrame({"timestamp": timestamps, "close": close, "volume": volume})
    return df


def build_features(df: pd.DataFrame, horizon: int, lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    for lag in range(1, lags + 1):
        work[f"lag_close_{lag}"] = work["close"].shift(lag)
        work[f"lag_return_{lag}"] = work["close"].pct_change(lag)
    work["volume"] = work["volume"].fillna(method="ffill").fillna(0.0)
    work["volume_change"] = work["volume"].pct_change().fillna(0.0)
    target = work["close"].shift(-horizon)
    features = work.drop(columns=["timestamp", "close"])
    valid = features.notna().all(axis=1) & target.notna()
    return features[valid], target[valid]


def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[GradientBoostingRegressor, TrainingResult]:
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    actual_dir = np.sign(y_test.values - X_test["lag_close_1"].values)
    pred_dir = np.sign(preds - X_test["lag_close_1"].values)
    directional_accuracy = float((actual_dir == pred_dir).mean())

    return model, TrainingResult(
        symbol="",
        horizon=0,
        current_price=float("nan"),
        predicted_price=float("nan"),
        r2=float(r2),
        rmse=float(rmse),
        directional_accuracy=directional_accuracy,
    )


def predict_next(model: GradientBoostingRegressor, df: pd.DataFrame, horizon: int, lags: int) -> Tuple[float, float]:
    latest = df.iloc[-1]
    features = {}
    for lag in range(1, lags + 1):
        features[f"lag_close_{lag}"] = df["close"].iloc[-lag]
        features[f"lag_return_{lag}"] = df["close"].pct_change(lag).iloc[-1]
    volume = df["volume"].fillna(method="ffill").iloc[-1]
    features["volume"] = volume
    features["volume_change"] = df["volume"].pct_change().fillna(0.0).iloc[-1]
    X_last = pd.DataFrame([features], columns=model.feature_names_in_)
    predicted_price = float(model.predict(X_last)[0])
    current_price = float(latest["close"])
    return current_price, predicted_price


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 supervised predictor")
    parser.add_argument("--csv", help="Path to historical CSV (optional)")
    parser.add_argument("--symbol", default="SYNTH", help="Ticker symbol for logging")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in steps")
    parser.add_argument("--lags", type=int, default=5, help="Number of lag features")
    parser.add_argument("--periods", type=int, default=2000, help="Synthetic periods when CSV is omitted")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = load_price_data(csv_path)
        symbol = args.symbol
        source = f"CSV ({csv_path})"
    else:
        symbol = args.symbol or "SYNTH"
        df = generate_synthetic_data(symbol, periods=args.periods)
        source = f"Synthetic random walk ({args.periods} rows)"
    features, target = build_features(df, args.horizon, args.lags)
    if len(features) < 50:
        raise RuntimeError("Not enough data after feature engineering.")
    model, metrics = train_model(features, target)
    current_price, predicted_price = predict_next(model, df, args.horizon, args.lags)
    metrics.symbol = symbol
    metrics.horizon = args.horizon
    metrics.current_price = current_price
    metrics.predicted_price = predicted_price

    print(f"[Source] {source}")
    print(f"[Symbol] {symbol}")
    print(f"[Samples] train/test split -> {int(len(features)*0.7)}/{len(features)-int(len(features)*0.7)}")
    print(f"[R2] {metrics.r2:.4f}")
    print(f"[RMSE] {metrics.rmse:.4f}")
    print(f"[Directional Accuracy] {metrics.directional_accuracy*100:.2f}%")
    print(f"[Current Price] {metrics.current_price:.2f}")
    print(f"[Predicted Next Price (horizon={metrics.horizon})] {metrics.predicted_price:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
