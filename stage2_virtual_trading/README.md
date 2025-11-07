# Stage 2 · Hybrid Virtual Trading

Stage 2 now ships with two complementary entry points:

1. **GUI Executor (`main.py`)** – manual mock-trading console that talks to the Kiwoom REST mock endpoints.
2. **Hybrid AI Trainer (`run_ai_training.py`)** – synthetic end-to-end workflow that mirrors the Stage‑A (selector) + Stage‑B (RL executor) pipeline.

## Features

- Configure mock App Key/Secret and send authenticated buy/sell/hold requests through the GUI.
- Fetch mock account snapshots (예수금/평가금/손익) directly from the GUI.
- Track virtual portfolio state (cash, market value, PnL) and review a trade log.
- Kick off the full Stage‑A/Stage‑B “Auto Learn” pipeline from the GUI with a single button (synthetic data by default).
- Generate synthetic OHLCV + indicator data, train a selector (LightGBM/XGBoost-style), and feed the top‑K tickers into a tick-level RL environment.
- Optional PPO training via `stable-baselines3`; automatic fallback to a deterministic baseline when the dependency is missing.

## Requirements

- Python 3.9+
- PyQt5 for the GUI.
- Optional:
  - `lightgbm` for faster selector training (otherwise uses `sklearn`).
  - `gymnasium` + `stable-baselines3` for RL training (otherwise uses a lightweight RandomAgent).

## Running the GUI

```bash
python stage2_virtual_trading/main.py
```

1. Enter mock App Key/Secret → click **모의투자 인증**.
2. Start the virtual broker with an initial cash balance.
3. Provide symbol/price/quantity and use **매수/매도/보유** to simulate orders.  
   Successful API submissions and errors are appended to the log panel.

## Running the Hybrid AI Trainer

```bash
python stage2_virtual_trading/run_ai_training.py --timesteps 100000 --ticks 60 --topk 20
```

Pipeline overview:

1. Generate/load universe data (synthetic by default).
2. Prepare scaled feature tensors with walk-forward splits.
3. Train the selector, compute risk-adjusted scores, and keep the top-K tickers.
4. Create tick-level environments for the selected tickers and launch PPO (or fallback) training.

Use `--help` to inspect additional switches (log level, timesteps, horizon, etc.). Replace the synthetic data generator with your production ETL by modifying `generate_synthetic_universe` or plugging in your dataset.
