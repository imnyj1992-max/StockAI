# Stage 4 · Supervised Learning Workbench

Stage 4 now mirrors the production checklist for supervised learning:

1. Connect to a mock brokerage account and surface its balance/holdings.
2. Load the virtual account plus the available symbol universe (from CSVs or synthetic data).
3. Choose the model family (RandomForest, GradientBoosting, LightGBM, XGBoost, ElasticNet, MLP, LSTM\*) and the stack of timeframes (1tick → 30min).
4. Kick off training:
   - Pick a random symbol every episode.
   - Load its full history from listing day to “now”, resample to each timeframe, and engineer leak-free lag features.
   - Train the selected models per timeframe, evaluate (R²/RMSE/directional accuracy) and persist the artefacts.
   - Loop until the requested number of episodes is finished.
5. Reload any saved model and stream real-time predictions while tracking cumulative accuracy.

\*LSTM support is enabled when `torch` is installed; otherwise it is skipped automatically.

---

## Requirements

```bash
pip install pandas numpy scikit-learn joblib
# optional accelerators / GUI
pip install lightgbm xgboost torch PyQt5
```

## Data layout

`train.py` can ingest:

- A `--data-root` directory that contains CSVs named `<SYMBOL>.csv`, `<SYMBOL>_<timeframe>.csv`, or nested as `<SYMBOL>/<timeframe>.csv`.
- A fallback synthetic generator (set `--synthetic-periods` to control the size).

Each CSV must expose `timestamp`/`date`, `close`, and optional `volume` columns. Example:

```
timestamp,open,high,low,close,volume
2024-01-02 09:00,70100,70200,70000,70150,10500
```

## GUI mode

Run the script without arguments (double-clicking on Windows also works):

```bash
python stage4_supervised_learning/train.py
# or explicitly
python stage4_supervised_learning/train.py gui
```

The GUI exposes two tabs:

- **학습**: select symbols, timeframes, model set, episodes, synthetic data length, seed, and mock-account credentials. Press **학습 시작** to run the multi-model pipeline and watch the log panel update in real time.
- **실시간 예측**: choose a saved model (direct `.joblib` file or model-dir + symbol + model-name), optional CSV/data-root, stream toggle, and tick limit. Press **예측 시작** to watch price forecasts plus accuracy stats accumulate in the log panel.

All CLI options remain available for automation; the GUI simply builds the same argument namespace.

## Training workflow (CLI)

```bash
python stage4_supervised_learning/train.py train \
  --data-root data/korea \
  --symbols 005930 000660 035720 \
  --timeframes 1tick 10tick 1min 5min 30min \
  --models random_forest lightgbm xgboost mlp lstm \
  --episodes 5 \
  --model-dir stage4_supervised_learning/artifacts
```

Key switches:

- `--app-key / --app-secret / --account-file`: mock-broker credentials and optional JSON snapshot.
- `--timeframes`: any mix of `Ntick` or `Nmin` values to resample and train per frequency.
- `--models`: choose the learner set; unavailable back-ends (e.g., LightGBM not installed) are skipped gracefully.
- `--min-samples`: guardrail for tiny datasets after feature engineering.
- `--model-dir`: destination for artefacts named `<symbol>/<model>_<timeframe>.joblib` with full metadata.

Each training episode logs the connected account, randomly selected symbol, evaluation metrics, and where the model was stored.

## Real-time prediction (CLI)

```bash
python stage4_supervised_learning/train.py predict \
  --model-dir stage4_supervised_learning/artifacts \
  --model-name random_forest \
  --symbol 005930 \
  --timeframe 1min \
  --limit 50 \
  --stream
```

The predictor loads the saved artefact (or use `--model-path` directly), reconstructs engineered features, and streams horizon-ahead forecasts. Each tick reports the previous price, prediction, actual price, rolling accuracy, and a final RMSE + cumulative accuracy summary. Use `--sleep` to throttle the loop to real-time cadence.

## Mock account payload (optional)

```json
{
  "account_id": "VIRTUAL-0001",
  "owner": "Mock User",
  "balance": 100000000,
  "currency": "KRW",
  "holdings": {
    "005930": 120,
    "000660": 45
  }
}
```

Passing this file via `--account-file` allows the CLI to echo realistic balances before each training cycle.
