from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from contextlib import redirect_stderr, redirect_stdout
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    QtCore = None
    QtGui = None
    QtWidgets = None


DEFAULT_SYMBOL_POOL = ["005930", "000660", "035420", "035720", "068270"]


@dataclass
class MockAccount:
    account_id: str
    owner: str
    balance: float
    currency: str = "KRW"
    holdings: Dict[str, float] = field(default_factory=dict)


@dataclass
class TimeframeSpec:
    label: str
    kind: str  # tick | time
    value: int
    pandas_freq: Optional[str] = None

    @staticmethod
    def parse(value: str) -> "TimeframeSpec":
        normalized = value.lower().strip()
        if normalized.endswith("tick"):
            size = int(normalized.replace("tick", "") or "1")
            return TimeframeSpec(label=f"{size}tick", kind="tick", value=size)
        if normalized.endswith("min"):
            size = int(normalized.replace("min", "") or "1")
            return TimeframeSpec(
                label=f"{size}min", kind="time", value=size, pandas_freq=f"{size}min"
            )
        raise ValueError(f"Unsupported timeframe: {value}")

    @property
    def file_key(self) -> str:
        return self.label.lower()

    def __str__(self) -> str:  # pragma: no cover
        return self.label


@dataclass
class ModelMetadata:
    symbol: str
    model_name: str
    timeframe: str
    horizon: int
    lags: int
    feature_columns: Sequence[str]
    seq_length: Optional[int] = None


@dataclass
class TrainingResult:
    symbol: str
    model_name: str
    timeframe: str
    samples: int
    r2: float
    rmse: float
    directional_accuracy: float
    current_price: float
    predicted_price: float
    artifact_path: Path


@dataclass
class FeatureDataset:
    tabular_X: pd.DataFrame
    tabular_y: pd.Series
    sequence_X: Optional[np.ndarray]
    sequence_y: Optional[np.ndarray]
    feature_columns: List[str]
    seq_length: int

    @property
    def tabular_size(self) -> int:
        return len(self.tabular_X)

    @property
    def sequence_size(self) -> int:
        return 0 if self.sequence_X is None else len(self.sequence_X)


def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("date")
    if not ts_col:
        raise ValueError("CSV must contain a 'timestamp' or 'date' column")
    if "close" not in cols:
        raise ValueError("CSV must contain a 'close' column")
    df = df.rename(columns={ts_col: "timestamp", cols["close"]: "close"})
    if "volume" in cols:
        df = df.rename(columns={cols["volume"]: "volume"})
    else:
        df["volume"] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_synthetic_data(symbol: str, periods: int = 2000, freq: str = "1min") -> pd.DataFrame:
    end_time = datetime.now().replace(second=0, microsecond=0)
    timestamps = pd.date_range(end=end_time, periods=periods, freq=freq)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    drift = 0.0002
    shock = rng.normal(drift, 0.01, size=periods)
    close = 100 * np.exp(np.cumsum(shock))
    volume = rng.integers(1_000, 15_000, size=periods)
    return pd.DataFrame({"timestamp": timestamps, "close": close, "volume": volume})


def prepare_timeframe_data(df: pd.DataFrame, timeframe: TimeframeSpec) -> pd.DataFrame:
    work = df.copy()
    work["volume"] = work["volume"].ffill().fillna(0.0)
    if timeframe.kind == "tick":
        group = np.arange(len(work)) // max(1, timeframe.value)
        agg = (
            work.assign(bucket=group)
            .groupby("bucket")
            .agg({"timestamp": "last", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index(drop=True)
        )
        return agg
    resampled = (
        work.set_index("timestamp")
        .resample(timeframe.pandas_freq)
        .agg({"close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return resampled


def build_tabular_features(df: pd.DataFrame, horizon: int, lags: int) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    for lag in range(1, lags + 1):
        work[f"lag_close_{lag}"] = work["close"].shift(lag)
        work[f"lag_return_{lag}"] = work["close"].pct_change(lag)
    work["volume"] = work["volume"].ffill().fillna(0.0)
    work["volume_change"] = work["volume"].pct_change().fillna(0.0)
    target = work["close"].shift(-horizon)
    features = work.drop(columns=["timestamp"] if "timestamp" in work.columns else [])
    features = features.drop(columns=["close"])
    valid = features.notna().all(axis=1) & target.notna()
    return features[valid], target[valid]


def build_sequence_dataset(df: pd.DataFrame, horizon: int, seq_length: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if seq_length <= 1:
        return None, None
    cols = ["close", "volume"]
    values = df[cols].values
    samples: List[np.ndarray] = []
    targets: List[float] = []
    for idx in range(seq_length, len(df) - horizon):
        window = values[idx - seq_length : idx]
        target = df["close"].iloc[idx + horizon]
        if np.isnan(window).any() or np.isnan(target):
            continue
        samples.append(window)
        targets.append(target)
    if not samples:
        return None, None
    return np.stack(samples), np.array(targets, dtype=np.float32)


def build_latest_feature_row(df: pd.DataFrame, columns: Sequence[str], lags: int) -> pd.DataFrame:
    if len(df) <= lags:
        raise RuntimeError("Not enough rows to build inference features.")
    row: Dict[str, float] = {}
    for lag in range(1, lags + 1):
        row[f"lag_close_{lag}"] = float(df["close"].iloc[-lag])
        row[f"lag_return_{lag}"] = float(df["close"].pct_change(lag).iloc[-1])
    volume = df["volume"].ffill().iloc[-1]
    row["volume"] = float(volume)
    row["volume_change"] = float(df["volume"].pct_change().fillna(0.0).iloc[-1])
    frame = pd.DataFrame([row])
    return frame[[c for c in columns if c in frame.columns]]


def build_latest_sequence(df: pd.DataFrame, seq_length: int) -> np.ndarray:
    if len(df) < seq_length:
        raise RuntimeError("Not enough rows to build inference sequence.")
    window = df[["close", "volume"]].tail(seq_length)
    return window.to_numpy(dtype=np.float32)[None, :, :]


class BaseModelAdapter:
    name: str
    requires_sequence: bool = False

    def fit(self, X: Any, y: Any) -> None:
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError


class SklearnRegressorAdapter(BaseModelAdapter):
    def __init__(self, name: str, estimator):
        self.name = name
        self.requires_sequence = False
        self.model = estimator

    def fit(self, X: Any, y: Any) -> None:
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class LightGBMAdapter(BaseModelAdapter):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        if lgb is None:
            raise RuntimeError("LightGBM is not installed.")
        self.name = name
        self.requires_sequence = False
        params = params or {}
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: Any, y: Any) -> None:
        self.model = lgb.LGBMRegressor(**self.model.get_params())
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class XGBoostAdapter(BaseModelAdapter):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        if xgb is None:
            raise RuntimeError("XGBoost is not installed.")
        self.name = name
        self.requires_sequence = False
        params = params or {}
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            **params,
        )

    def fit(self, X: Any, y: Any) -> None:
        self.model = xgb.XGBRegressor(**self.model.get_params())
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class TorchLSTMAdapter(BaseModelAdapter):
    def __init__(
        self,
        input_size: int,
        name: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        if torch is None or nn is None or DataLoader is None:
            raise RuntimeError("PyTorch is not installed.")
        self.name = name
        self.requires_sequence = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = _LstmRegressorNet(input_size, hidden_size, num_layers, dropout)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for features, target in loader:
                optimizer.zero_grad()
                preds = self.model(features)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            preds = self.model(tensor).cpu().numpy()
        return preds


class _LstmRegressorNet(nn.Module):  # pragma: no cover - helper only
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.fc(last_step).squeeze(-1)


@dataclass
class ModelFactory:
    name: str
    requires_sequence: bool
    available: bool
    builder: Any
    note: Optional[str] = None

    def create(self, sequence_input_size: Optional[int] = None) -> BaseModelAdapter:
        return self.builder(sequence_input_size)


def create_model_factories() -> Dict[str, ModelFactory]:
    factories: Dict[str, ModelFactory] = {
        "gradient_boosting": ModelFactory(
            name="gradient_boosting",
            requires_sequence=False,
            available=True,
            builder=lambda _: SklearnRegressorAdapter(
                "gradient_boosting", GradientBoostingRegressor(random_state=42)
            ),
        ),
        "random_forest": ModelFactory(
            name="random_forest",
            requires_sequence=False,
            available=True,
            builder=lambda _: SklearnRegressorAdapter(
                "random_forest",
                RandomForestRegressor(
                    n_estimators=400, max_depth=8, n_jobs=-1, random_state=42
                ),
            ),
        ),
        "elasticnet": ModelFactory(
            name="elasticnet",
            requires_sequence=False,
            available=True,
            builder=lambda _: SklearnRegressorAdapter(
                "elasticnet",
                ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=5000, random_state=42),
            ),
        ),
        "mlp": ModelFactory(
            name="mlp",
            requires_sequence=False,
            available=True,
            builder=lambda _: SklearnRegressorAdapter(
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    max_iter=400,
                    random_state=42,
                ),
            ),
        ),
    }
    if lgb is not None:
        factories["lightgbm"] = ModelFactory(
            name="lightgbm",
            requires_sequence=False,
            available=True,
            builder=lambda _: LightGBMAdapter("lightgbm", {"n_estimators": 1000, "learning_rate": 0.02}),
        )
    else:
        factories["lightgbm"] = ModelFactory(
            name="lightgbm",
            requires_sequence=False,
            available=False,
            builder=lambda _: (_ for _ in ()).throw(RuntimeError("LightGBM missing")),
            note="lightgbm package not installed",
        )
    if xgb is not None:
        factories["xgboost"] = ModelFactory(
            name="xgboost",
            requires_sequence=False,
            available=True,
            builder=lambda _: XGBoostAdapter("xgboost", {"n_estimators": 800, "learning_rate": 0.03}),
        )
    else:
        factories["xgboost"] = ModelFactory(
            name="xgboost",
            requires_sequence=False,
            available=False,
            builder=lambda _: (_ for _ in ()).throw(RuntimeError("XGBoost missing")),
            note="xgboost package not installed",
        )
    if torch is not None:
        factories["lstm"] = ModelFactory(
            name="lstm",
            requires_sequence=True,
            available=True,
            builder=lambda seq_size: TorchLSTMAdapter(input_size=seq_size or 2),
        )
    else:
        factories["lstm"] = ModelFactory(
            name="lstm",
            requires_sequence=True,
            available=False,
            builder=lambda _: (_ for _ in ()).throw(RuntimeError("PyTorch missing")),
            note="torch package not installed",
        )
    return factories


class MockAccountConnector:
    def __init__(self, app_key: str, app_secret: str, account_file: Optional[Path] = None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_file = account_file

    def connect(self) -> MockAccount:
        if not self.app_key or not self.app_secret:
            raise ValueError("App key/secret are required for mock account connection.")
        data = self._load_account_payload()
        return MockAccount(
            account_id=data.get("account_id", "VIRTUAL-0001"),
            owner=data.get("owner", "Mock User"),
            balance=float(data.get("balance", 100_000_000)),
            currency=data.get("currency", "KRW"),
            holdings=data.get("holdings", {}),
        )

    def _load_account_payload(self) -> Dict[str, Any]:
        if self.account_file and self.account_file.exists():
            with self.account_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "account_id": "VIRTUAL-0001",
            "owner": "Mock User",
            "balance": 100_000_000,
            "currency": "KRW",
            "holdings": {},
        }


class SymbolDataRepository:
    def __init__(
        self,
        data_root: Optional[Path],
        fallback_symbols: Sequence[str],
        synthetic_periods: int,
    ):
        self.data_root = data_root if data_root and data_root.exists() else None
        self.synthetic_periods = synthetic_periods
        self._fallback_symbols = [s.upper() for s in fallback_symbols]
        self._base_cache: Dict[str, pd.DataFrame] = {}
        self._symbols = self._discover_symbols()

    def _discover_symbols(self) -> List[str]:
        if not self.data_root:
            return self._fallback_symbols
        symbols: set[str] = set()
        for csv_path in self.data_root.rglob("*.csv"):
            name = csv_path.stem.split("_")[0].upper()
            if name:
                symbols.add(name)
        for child in self.data_root.iterdir():
            if child.is_dir():
                symbols.add(child.name.upper())
        symbols.update(self._fallback_symbols)
        return sorted(symbols)

    def available_symbols(self) -> List[str]:
        return self._symbols

    def load(self, symbol: str, timeframe: TimeframeSpec) -> pd.DataFrame:
        symbol = symbol.upper()
        df = self._load_from_disk(symbol, timeframe)
        if df is not None:
            return df
        return self._load_base(symbol)

    def _candidate_paths(self, symbol: str, timeframe: TimeframeSpec) -> Iterable[Path]:
        if not self.data_root:
            return []
        yield self.data_root / symbol / f"{timeframe.file_key}.csv"
        yield self.data_root / symbol / f"{symbol}_{timeframe.file_key}.csv"
        yield self.data_root / f"{symbol}_{timeframe.file_key}.csv"
        yield self.data_root / f"{symbol}.csv"

    def _load_from_disk(self, symbol: str, timeframe: TimeframeSpec) -> Optional[pd.DataFrame]:
        for candidate in self._candidate_paths(symbol, timeframe):
            if candidate.exists():
                try:
                    return load_price_data(candidate)
                except Exception:
                    continue
        return None

    def _load_base(self, symbol: str) -> pd.DataFrame:
        if symbol in self._base_cache:
            return self._base_cache[symbol].copy()
        df = generate_synthetic_data(symbol, periods=self.synthetic_periods)
        self._base_cache[symbol] = df
        return df.copy()


class SupervisedTrainingPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.account_connector = MockAccountConnector(
            args.app_key, args.app_secret, args.account_file
        )
        data_root = Path(args.data_root).expanduser() if args.data_root else None
        self.data_repo = SymbolDataRepository(
            data_root=data_root,
            fallback_symbols=args.symbols or DEFAULT_SYMBOL_POOL,
            synthetic_periods=args.synthetic_periods,
        )
        self.timeframes = [TimeframeSpec.parse(tf) for tf in args.timeframes]
        self.model_factories = create_model_factories()
        self.model_dir = Path(args.model_dir).expanduser()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = args.train_ratio
        self.min_samples = args.min_samples
        self.horizon = args.horizon
        self.lags = args.lags
        self.seq_length = args.seq_length or max(3, args.lags)
        random.seed(args.seed)
        np.random.seed(args.seed)

    def run(self) -> None:
        account = self.account_connector.connect()
        self._log_account(account)
        symbols = self.data_repo.available_symbols()
        if not symbols:
            raise RuntimeError("No symbols available for training.")
        for episode in range(1, self.args.episodes + 1):
            symbol = random.choice(symbols)
            print(f"\n[Episode {episode}/{self.args.episodes}] ▶ Symbol picked: {symbol}")
            for timeframe in self.timeframes:
                df_raw = self.data_repo.load(symbol, timeframe)
                df = prepare_timeframe_data(df_raw, timeframe)
                dataset = self._build_dataset(df)
                self._train_all_models(symbol, timeframe, dataset, df)

    def _log_account(self, account: MockAccount) -> None:
        holdings = ", ".join(f"{sym}:{qty}" for sym, qty in account.holdings.items()) or "-"
        print("[Account] Connected to mock broker")
        print(
            f"  · Account: {account.account_id} ({account.owner}) / "
            f"Balance: {account.balance:,.0f} {account.currency}"
        )
        print(f"  · Holdings: {holdings}")

    def _build_dataset(self, df: pd.DataFrame) -> FeatureDataset:
        features, target = build_tabular_features(df, self.horizon, self.lags)
        seq_X, seq_y = build_sequence_dataset(df, self.horizon, self.seq_length)
        return FeatureDataset(
            tabular_X=features,
            tabular_y=target,
            sequence_X=seq_X,
            sequence_y=seq_y,
            feature_columns=list(features.columns),
            seq_length=self.seq_length,
        )

    def _train_all_models(
        self,
        symbol: str,
        timeframe: TimeframeSpec,
        dataset: FeatureDataset,
        df: pd.DataFrame,
    ) -> None:
        for model_name in self.args.models:
            factory = self.model_factories.get(model_name.lower())
            if not factory:
                print(f"  - [{model_name}] skipped (unknown model)")
                continue
            if not factory.available:
                note = f" ({factory.note})" if factory.note else ""
                print(f"  - [{model_name}] skipped{note}")
                continue
            if factory.requires_sequence and dataset.sequence_size < self.min_samples:
                print(f"  - [{model_name}] skipped (sequence samples too small)")
                continue
            if not factory.requires_sequence and dataset.tabular_size < self.min_samples:
                print(f"  - [{model_name}] skipped (samples too small)")
                continue
            adapter = factory.create(
                sequence_input_size=None
                if dataset.sequence_X is None
                else dataset.sequence_X.shape[-1]
            )
            result = self._fit_single_model(adapter, dataset, df, symbol, timeframe)
            self._print_training_result(result)

    def _fit_single_model(
        self,
        adapter: BaseModelAdapter,
        dataset: FeatureDataset,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeframeSpec,
    ) -> TrainingResult:
        if adapter.requires_sequence:
            X, y = dataset.sequence_X, dataset.sequence_y
        else:
            X, y = dataset.tabular_X, dataset.tabular_y
        total_samples = len(X)
        if total_samples < 2:
            raise RuntimeError("Not enough samples to train.")
        split_idx = int(total_samples * self.train_ratio)
        split_idx = max(1, min(total_samples - 1, split_idx))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        adapter.fit(X_train, y_train)
        preds = adapter.predict(X_test)
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        directional_accuracy = self._compute_directional_accuracy(
            adapter, X_test, y_test, dataset
        )
        current_price = float(df["close"].iloc[-1])
        predicted_price = self._infer_latest_price(adapter, df, dataset)
        metadata = ModelMetadata(
            symbol=symbol,
            model_name=adapter.name,
            timeframe=timeframe.label,
            horizon=self.horizon,
            lags=self.lags,
            feature_columns=dataset.feature_columns,
            seq_length=dataset.seq_length if adapter.requires_sequence else None,
        )
        artifact_path = self._save_model(adapter, metadata)
        return TrainingResult(
            symbol=symbol,
            model_name=adapter.name,
            timeframe=timeframe.label,
            samples=total_samples,
            r2=float(r2),
            rmse=float(rmse),
            directional_accuracy=float(directional_accuracy),
            current_price=current_price,
            predicted_price=predicted_price,
            artifact_path=artifact_path,
        )

    def _compute_directional_accuracy(
        self,
        adapter: BaseModelAdapter,
        X_test: Any,
        y_test: Any,
        dataset: FeatureDataset,
    ) -> float:
        if len(y_test) == 0:
            return float("nan")
        if adapter.requires_sequence:
            prev_close = X_test[:, -1, 0]
        else:
            prev_close = X_test["lag_close_1"].values
        actual_dir = np.sign(y_test - prev_close)
        preds = adapter.predict(X_test)
        pred_dir = np.sign(preds - prev_close)
        return float(np.mean(actual_dir == pred_dir))

    def _infer_latest_price(
        self,
        adapter: BaseModelAdapter,
        df: pd.DataFrame,
        dataset: FeatureDataset,
    ) -> float:
        if adapter.requires_sequence:
            seq = build_latest_sequence(df, dataset.seq_length)
            return float(adapter.predict(seq)[0])
        feature_row = build_latest_feature_row(df, dataset.feature_columns, self.lags)
        aligned = feature_row.reindex(columns=dataset.feature_columns, fill_value=0.0)
        return float(adapter.predict(aligned)[0])

    def _save_model(self, adapter: BaseModelAdapter, metadata: ModelMetadata) -> Path:
        symbol_dir = self.model_dir / metadata.symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{metadata.model_name}_{metadata.timeframe}.joblib"
        payload = {"adapter": adapter, "metadata": asdict(metadata)}
        path = symbol_dir / filename
        joblib.dump(payload, path)
        return path

    def _print_training_result(self, result: TrainingResult) -> None:
        print(
            f"  - [{result.model_name}/{result.timeframe}] "
            f"samples={result.samples} "
            f"R2={result.r2:.3f} RMSE={result.rmse:.2f} "
            f"DirAcc={result.directional_accuracy*100:.1f}% "
            f"Now={result.current_price:.2f} Next={result.predicted_price:.2f} "
            f"=> saved {result.artifact_path}"
        )


class RealTimePredictor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        data_root = Path(args.data_root).expanduser() if args.data_root else None
        self.data_repo = SymbolDataRepository(
            data_root=data_root,
            fallback_symbols=args.symbols or DEFAULT_SYMBOL_POOL,
            synthetic_periods=args.synthetic_periods,
        )
        self.model_payload = self._load_model_payload()
        self.adapter: BaseModelAdapter = self.model_payload["adapter"]
        self.metadata = ModelMetadata(**self.model_payload["metadata"])
        self.timeframe = TimeframeSpec.parse(args.timeframe or self.metadata.timeframe)

    def _resolve_model_path(self) -> Path:
        if self.args.model_path:
            return Path(self.args.model_path).expanduser()
        if not (self.args.model_dir and self.args.symbol and self.args.model_name):
            raise ValueError("Provide --model-path or model-dir + symbol + model-name.")
        symbol_dir = Path(self.args.model_dir).expanduser() / self.args.symbol
        if self.args.timeframe:
            tf_label = TimeframeSpec.parse(self.args.timeframe).label
            path = symbol_dir / f"{self.args.model_name}_{tf_label}.joblib"
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        matches = sorted(symbol_dir.glob(f"{self.args.model_name}_*.joblib"))
        if not matches:
            raise FileNotFoundError(
                f"No artifacts matching {self.args.model_name}_*.joblib in {symbol_dir}"
            )
        return matches[0]

    def _load_model_payload(self) -> Dict[str, Any]:
        path = self._resolve_model_path()
        payload = joblib.load(path)
        if "adapter" not in payload or "metadata" not in payload:
            raise RuntimeError("Model artifact is missing metadata.")
        return payload

    def _load_price_frame(self) -> pd.DataFrame:
        if self.args.csv:
            csv_path = Path(self.args.csv).expanduser()
            return load_price_data(csv_path)
        symbol = self.args.symbol or self.metadata.symbol
        df_raw = self.data_repo.load(symbol, self.timeframe)
        return df_raw

    def run(self) -> None:
        df_raw = self._load_price_frame()
        df = prepare_timeframe_data(df_raw, self.timeframe)
        lag = self.metadata.lags
        seq_len = self.metadata.seq_length or max(lag, 3)
        horizon = self.metadata.horizon
        prices: List[float] = []
        preds: List[float] = []
        correct = 0
        iterations = 0
        limit = self.args.limit or len(df)
        for idx in range(max(lag, seq_len), len(df) - horizon):
            if iterations >= limit:
                break
            window = df.iloc[: idx + 1]
            if self.adapter.requires_sequence:
                features = build_latest_sequence(window, seq_len)
                prev_price = float(features[0, -1, 0])
            else:
                features = build_latest_feature_row(
                    window, self.metadata.feature_columns, lag
                )
                features = features.reindex(
                    columns=self.metadata.feature_columns, fill_value=0.0
                )
                prev_price = float(window["close"].iloc[-2])
            prediction = float(self.adapter.predict(features)[0])
            actual = float(df["close"].iloc[idx + horizon])
            prices.append(actual)
            preds.append(prediction)
            if np.sign(actual - prev_price) == np.sign(prediction - prev_price):
                correct += 1
            iterations += 1
            if self.args.stream:
                print(
                    f"[{iterations}] prev={prev_price:.2f} pred={prediction:.2f} "
                    f"actual={actual:.2f} acc={correct/iterations*100:.1f}%"
                )
            if self.args.sleep > 0:
                time.sleep(self.args.sleep)
        if iterations == 0:
            raise RuntimeError("Not enough samples to run prediction stream.")
        rmse = float(np.sqrt(mean_squared_error(prices, preds)))
        accuracy = correct / iterations
        print("\n[Real-time Prediction]")
        print(f"  · Samples processed: {iterations}")
        print(f"  · Cumulative accuracy: {accuracy*100:.2f}%")
        print(f"  · RMSE: {rmse:.4f}")
        print(f"  · Last actual: {prices[-1]:.2f} / Last predicted: {preds[-1]:.2f}")


def run_training_job(args: argparse.Namespace) -> None:
    pipeline = SupervisedTrainingPipeline(args)
    pipeline.run()


def run_prediction_job(args: argparse.Namespace) -> None:
    predictor = RealTimePredictor(args)
    predictor.run()


def _split_user_list(raw: str) -> List[str]:
    if not raw:
        return []
    items: List[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        token = chunk.strip()
        if token:
            items.append(token)
    return items


def _text_or_none(value: str) -> Optional[str]:
    text = value.strip()
    return text or None


def _path_or_none(value: str) -> Optional[Path]:
    text = _text_or_none(value)
    return Path(text).expanduser() if text else None


if QtWidgets is not None and QtCore is not None and QtGui is not None:

    class _SignalWriter(io.TextIOBase):  # pragma: no cover - GUI helper
        def __init__(self, signal):
            super().__init__()
            self.signal = signal

        def write(self, text: str) -> int:
            if text:
                self.signal.emit(text)
            return len(text)

        def flush(self) -> None:
            pass


    class Stage4Worker(QtCore.QThread):  # pragma: no cover - GUI helper
        log = QtCore.pyqtSignal(str)
        completed = QtCore.pyqtSignal(bool, str)

        def __init__(self, target, *args, **kwargs):
            super().__init__()
            self._target = target
            self._args = args
            self._kwargs = kwargs

        def run(self) -> None:
            writer = _SignalWriter(self.log)
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    self._target(*self._args, **self._kwargs)
            except Exception:
                self.completed.emit(False, traceback.format_exc())
            else:
                self.completed.emit(True, "완료")


    class Stage4MainWindow(QtWidgets.QMainWindow):  # pragma: no cover - GUI helper
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Stage 4 · Supervised Learning")
            self.worker: Optional[Stage4Worker] = None
            self._build_ui()

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)
            self.tabs = QtWidgets.QTabWidget()
            layout.addWidget(self.tabs)
            self._init_train_tab()
            self._init_predict_tab()
            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setReadOnly(True)
            layout.addWidget(self.log_view)
            self.statusBar().showMessage("준비 완료")

        # --- Train tab ---
        def _init_train_tab(self) -> None:
            tab = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab)

            self.train_data_root = QtWidgets.QLineEdit()
            self.train_data_root.setPlaceholderText("예: data/korea")
            form.addRow("데이터 폴더", self.train_data_root)

            self.train_symbols = QtWidgets.QLineEdit("005930,000660,035720")
            form.addRow("심볼 목록", self.train_symbols)

            self.train_timeframes = QtWidgets.QLineEdit("1tick,10tick,1min,5min,30min")
            form.addRow("타임프레임", self.train_timeframes)

            self.train_models = QtWidgets.QLineEdit(
                "random_forest,lightgbm,xgboost,elasticnet,mlp,lstm"
            )
            form.addRow("모델", self.train_models)

            self.train_model_dir = QtWidgets.QLineEdit("stage4_supervised_learning/artifacts")
            form.addRow("모델 저장 경로", self.train_model_dir)

            self.train_episodes = QtWidgets.QSpinBox()
            self.train_episodes.setRange(1, 1000)
            self.train_episodes.setValue(2)
            form.addRow("에피소드 수", self.train_episodes)

            self.train_horizon = QtWidgets.QSpinBox()
            self.train_horizon.setRange(1, 60)
            self.train_horizon.setValue(1)
            form.addRow("예측 호라이즌", self.train_horizon)

            self.train_lags = QtWidgets.QSpinBox()
            self.train_lags.setRange(1, 60)
            self.train_lags.setValue(5)
            form.addRow("Lag 개수", self.train_lags)

            self.train_seq_length = QtWidgets.QSpinBox()
            self.train_seq_length.setRange(0, 500)
            self.train_seq_length.setSpecialValueText("자동")
            self.train_seq_length.setValue(0)
            form.addRow("시퀀스 길이(LSTM)", self.train_seq_length)

            self.train_ratio = QtWidgets.QDoubleSpinBox()
            self.train_ratio.setRange(0.5, 0.95)
            self.train_ratio.setSingleStep(0.05)
            self.train_ratio.setValue(0.75)
            form.addRow("Train Ratio", self.train_ratio)

            self.train_min_samples = QtWidgets.QSpinBox()
            self.train_min_samples.setRange(20, 100000)
            self.train_min_samples.setValue(200)
            form.addRow("최소 샘플 수", self.train_min_samples)

            self.train_synth_periods = QtWidgets.QSpinBox()
            self.train_synth_periods.setRange(100, 100000)
            self.train_synth_periods.setValue(4000)
            form.addRow("Synthetic Rows", self.train_synth_periods)

            self.train_seed = QtWidgets.QSpinBox()
            self.train_seed.setRange(1, 2_000_000_000)
            self.train_seed.setValue(42)
            form.addRow("Seed", self.train_seed)

            self.train_app_key = QtWidgets.QLineEdit("demo-app-key")
            form.addRow("App Key", self.train_app_key)

            self.train_app_secret = QtWidgets.QLineEdit("demo-app-secret")
            self.train_app_secret.setEchoMode(QtWidgets.QLineEdit.Password)
            form.addRow("App Secret", self.train_app_secret)

            self.train_account_file = QtWidgets.QLineEdit()
            self.train_account_file.setPlaceholderText("mock_account.json (선택)")
            form.addRow("계좌 JSON", self.train_account_file)

            self.train_button = QtWidgets.QPushButton("학습 시작")
            self.train_button.clicked.connect(self.start_training)
            form.addRow("", self.train_button)

            self.tabs.addTab(tab, "학습")

        # --- Predict tab ---
        def _init_predict_tab(self) -> None:
            tab = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab)

            self.predict_model_path = QtWidgets.QLineEdit()
            self.predict_model_path.setPlaceholderText("직접 모델 경로 (선택)")
            form.addRow("모델 파일", self.predict_model_path)

            self.predict_model_dir = QtWidgets.QLineEdit("stage4_supervised_learning/artifacts")
            form.addRow("모델 폴더", self.predict_model_dir)

            self.predict_model_name = QtWidgets.QLineEdit("random_forest")
            form.addRow("모델 이름", self.predict_model_name)

            self.predict_symbol = QtWidgets.QLineEdit("005930")
            form.addRow("심볼", self.predict_symbol)

            self.predict_timeframe = QtWidgets.QLineEdit("1min")
            form.addRow("타임프레임", self.predict_timeframe)

            self.predict_data_root = QtWidgets.QLineEdit()
            form.addRow("데이터 폴더", self.predict_data_root)

            self.predict_symbols = QtWidgets.QLineEdit("005930,000660,035720")
            form.addRow("대체 심볼", self.predict_symbols)

            self.predict_csv = QtWidgets.QLineEdit()
            self.predict_csv.setPlaceholderText("직접 CSV (선택)")
            form.addRow("CSV 경로", self.predict_csv)

            self.predict_limit = QtWidgets.QSpinBox()
            self.predict_limit.setRange(0, 100000)
            self.predict_limit.setValue(0)
            self.predict_limit.setSpecialValueText("전체")
            form.addRow("예측 스텝 제한", self.predict_limit)

            self.predict_synth_periods = QtWidgets.QSpinBox()
            self.predict_synth_periods.setRange(100, 100000)
            self.predict_synth_periods.setValue(4000)
            form.addRow("Synthetic Rows", self.predict_synth_periods)

            self.predict_stream = QtWidgets.QCheckBox("Streaming 로그 출력")
            self.predict_stream.setChecked(True)
            form.addRow("", self.predict_stream)

            self.predict_sleep = QtWidgets.QDoubleSpinBox()
            self.predict_sleep.setRange(0.0, 5.0)
            self.predict_sleep.setSingleStep(0.1)
            self.predict_sleep.setValue(0.0)
            form.addRow("Tick 간 지연(초)", self.predict_sleep)

            self.predict_button = QtWidgets.QPushButton("예측 시작")
            self.predict_button.clicked.connect(self.start_prediction)
            form.addRow("", self.predict_button)

            self.tabs.addTab(tab, "실시간 예측")

        # --- Shared helpers ---
        def append_log(self, text: str) -> None:
            self.log_view.moveCursor(QtGui.QTextCursor.End)
            self.log_view.insertPlainText(text)
            self.log_view.ensureCursorVisible()

        def _set_buttons_enabled(self, enabled: bool) -> None:
            self.train_button.setEnabled(enabled)
            self.predict_button.setEnabled(enabled)

        def _start_worker(self, target, *args, **kwargs) -> None:
            if self.worker and self.worker.isRunning():
                QtWidgets.QMessageBox.warning(self, "실행 중", "다른 작업이 아직 진행 중입니다.")
                return
            self.log_view.clear()
            self.worker = Stage4Worker(target, *args, **kwargs)
            self.worker.log.connect(self.append_log)
            self.worker.completed.connect(self._on_worker_finished)
            self._set_buttons_enabled(False)
            self.statusBar().showMessage("작업 실행 중...")
            self.worker.start()

        def _on_worker_finished(self, success: bool, message: str) -> None:
            self._set_buttons_enabled(True)
            self.statusBar().showMessage("완료" if success else "에러", 5000)
            if message:
                self.append_log("\n" + message)
            if not success:
                QtWidgets.QMessageBox.critical(self, "오류", message)

        def closeEvent(self, event) -> None:
            if self.worker and self.worker.isRunning():
                QtWidgets.QMessageBox.warning(
                    self, "실행 중", "작업이 끝날 때까지 기다려 주세요."
                )
                event.ignore()
                return
            super().closeEvent(event)

        def start_training(self) -> None:
            try:
                args = self._collect_training_args()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "입력 오류", str(exc))
                return
            self._start_worker(run_training_job, args)

        def _collect_training_args(self) -> argparse.Namespace:
            symbols = _split_user_list(self.train_symbols.text())
            timeframes = _split_user_list(self.train_timeframes.text()) or ["1min"]
            models = _split_user_list(self.train_models.text()) or ["random_forest"]
            seq_len = self.train_seq_length.value()
            seq_len_value = None if seq_len == 0 else seq_len
            account_path = _path_or_none(self.train_account_file.text())
            args = argparse.Namespace(
                command="train",
                data_root=_text_or_none(self.train_data_root.text()),
                symbols=symbols or None,
                timeframes=timeframes,
                models=models,
                episodes=self.train_episodes.value(),
                horizon=self.train_horizon.value(),
                lags=self.train_lags.value(),
                seq_length=seq_len_value,
                train_ratio=self.train_ratio.value(),
                min_samples=self.train_min_samples.value(),
                model_dir=_text_or_none(self.train_model_dir.text())
                or "stage4_supervised_learning/artifacts",
                synthetic_periods=self.train_synth_periods.value(),
                seed=self.train_seed.value(),
                app_key=self.train_app_key.text().strip() or "demo-app-key",
                app_secret=self.train_app_secret.text().strip() or "demo-app-secret",
                account_file=account_path,
            )
            return args

        def start_prediction(self) -> None:
            try:
                args = self._collect_prediction_args()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "입력 오류", str(exc))
                return
            self._start_worker(run_prediction_job, args)

        def _collect_prediction_args(self) -> argparse.Namespace:
            model_path = _text_or_none(self.predict_model_path.text())
            model_dir = _text_or_none(self.predict_model_dir.text())
            model_name = _text_or_none(self.predict_model_name.text())
            symbol = _text_or_none(self.predict_symbol.text())
            if not model_path and not (model_dir and model_name and symbol):
                raise ValueError("모델 파일 또는 (모델 폴더 + 모델명 + 심볼)이 필요합니다.")
            limit_value = self.predict_limit.value()
            limit = None if limit_value == 0 else limit_value
            args = argparse.Namespace(
                command="predict",
                model_path=model_path,
                model_dir=model_dir,
                model_name=model_name,
                symbol=symbol,
                timeframe=_text_or_none(self.predict_timeframe.text()),
                data_root=_text_or_none(self.predict_data_root.text()),
                symbols=_split_user_list(self.predict_symbols.text()) or None,
                csv=_text_or_none(self.predict_csv.text()),
                limit=limit,
                synthetic_periods=self.predict_synth_periods.value(),
                stream=self.predict_stream.isChecked(),
                sleep=self.predict_sleep.value(),
            )
            return args


else:  # pragma: no cover - GUI helper fallback
    Stage4Worker = None
    Stage4MainWindow = None


def launch_gui() -> None:
    if QtWidgets is None or QtCore is None:
        raise RuntimeError("PyQt5 미설치: `pip install PyQt5` 후 다시 시도하세요.")
    app = QtWidgets.QApplication.instance()
    should_cleanup = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
        should_cleanup = True
    window = Stage4MainWindow()
    window.resize(960, 720)
    window.show()
    app.exec_()
    if should_cleanup:
        del app


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 supervised learning workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run supervised multi-model training.")
    train_parser.add_argument("--data-root", help="Directory with CSV files per symbol.")
    train_parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol universe (defaults to discovery + synthetic).",
    )
    train_parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1tick", "10tick", "1min", "5min", "10min", "30min"],
        help="List of timeframes to iterate.",
    )
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "lightgbm", "xgboost", "elasticnet", "mlp", "lstm"],
        help="Model names to train.",
    )
    train_parser.add_argument("--episodes", type=int, default=2, help="Number of training episodes.")
    train_parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon.")
    train_parser.add_argument("--lags", type=int, default=5, help="Number of lag features.")
    train_parser.add_argument("--seq-length", type=int, help="Sequence length for LSTM (defaults to lags).")
    train_parser.add_argument("--train-ratio", type=float, default=0.75, help="Train/test split ratio.")
    train_parser.add_argument("--min-samples", type=int, default=200, help="Minimum samples per dataset.")
    train_parser.add_argument("--model-dir", default="stage4_supervised_learning/artifacts", help="Where to store trained models.")
    train_parser.add_argument("--synthetic-periods", type=int, default=4000, help="Synthetic periods when data missing.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument("--app-key", default="demo-app-key", help="Mock app key.")
    train_parser.add_argument("--app-secret", default="demo-app-secret", help="Mock app secret.")
    train_parser.add_argument("--account-file", type=Path, help="Optional mock account JSON.")

    predict_parser = subparsers.add_parser("predict", help="Load saved model and stream predictions.")
    predict_parser.add_argument("--model-path", help="Direct path to saved artifact.")
    predict_parser.add_argument("--model-dir", help="Directory containing artifacts.")
    predict_parser.add_argument("--model-name", help="Model name (used with model-dir).")
    predict_parser.add_argument("--symbol", help="Symbol to evaluate (defaults to metadata).")
    predict_parser.add_argument("--timeframe", help="Timeframe label (defaults to metadata).")
    predict_parser.add_argument("--data-root", help="Optional data directory.")
    predict_parser.add_argument("--symbols", nargs="+", default=None, help="Fallback symbols for synthetic data.")
    predict_parser.add_argument("--csv", help="Direct CSV for prediction stream.")
    predict_parser.add_argument("--limit", type=int, help="Limit number of prediction steps.")
    predict_parser.add_argument("--synthetic-periods", type=int, default=4000, help="Synthetic periods for fallback data.")
    predict_parser.add_argument("--stream", action="store_true", help="Print every prediction step.")
    predict_parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between steps.")
    subparsers.add_parser("gui", help="Launch the interactive GUI.")

    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    if not argv:
        try:
            launch_gui()
            return 0
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    args = parse_args(argv)
    if args.command == "gui":
        try:
            launch_gui()
            return 0
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.command == "train":
        run_training_job(args)
        return 0
    if args.command == "predict":
        run_prediction_job(args)
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
