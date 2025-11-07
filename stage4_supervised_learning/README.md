# Stage 4 · Supervised Price Prediction

Stage 4 skips the separate performance validation step and jumps straight into supervised learning. The goal is to feed the full historical time-series for a single symbol, train a regression model that predicts the next-period price, and then surface both the model accuracy and the current/predicted prices.

## Features

- Loads per-symbol historical data from CSV (any frequency – minutes, hours, days) as long as a close column exists.
- Automatically engineers lag features (returns and prices) with leak-free train/test split based on time.
- Trains a Gradient Boosting regressor (scikit-learn) and reports:
  - Regression R² (goodness of fit)
  - Directional accuracy (percentage of correct up/down predictions)
  - Current close price & predicted next close
- CLI-driven so you can integrate it into automated workflows.

## Requirements

`
pip install pandas numpy scikit-learn
`

## Data format

Input CSV must include at least:

- 	imestamp (or date) column parsable by pandas
- close price column
- Optional olume column (used as an extra feature when present)

Example (minute bars):

`
timestamp,open,high,low,close,volume
2024-01-02 09:00,70100,70200,70000,70150,10500
...
`

## Usage

`
python stage4_supervised_learning/train.py --csv data/005930_minute.csv --symbol 005930 --horizon 1
`

- --csv: Path to CSV containing the historical bars.
- --symbol: Identifier for logging purposes.
- --horizon: Number of steps ahead to predict (default 1).

Output includes model metrics plus current/predicted prices.

## Extending

- Replace Gradient Boosting with any regressor/classifier of your choice (LightGBM, CatBoost, deep nets, etc.).
- Hook the trainer into your data pipeline to retrain on schedule and persist the model artefact.
- Feed the predicted price into the Stage 2 mock trading or Stage 3 RL executor for hybrid workflows.
