# Stage 2 – Virtual Trading Test

Stage 2 builds a self-contained virtual trading simulator so that the GUI and
basic trade flow can be tested without access to Kiwoom OpenAPI+.  The
simulator focuses on cash-based accounting using an initial `cash_init` value
and user-entered trade prices.

## Features

- Configure an initial virtual cash balance.
- Enter manual trades (buy, sell, hold) for arbitrary symbols.
- Track current cash, market value, and aggregate profit & loss.
- View per-symbol positions in a sortable table and review the trade log.

## Requirements

- Windows environment is recommended for consistency with later stages, but the
  virtual simulator does not depend on Kiwoom-specific components.
- Python 3.9+
- `PyQt5`

## How to Run

```bash
python stage2_virtual_trading/main.py
```

The window enables trade buttons after the initial cash value is confirmed.
Input a symbol (e.g. `005930`), price, and quantity, then choose **매수** or
**매도** to simulate trades.  The **관망** button records a hold decision for the
current symbol.
