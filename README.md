# FINM 25000 Homework 3

Machine-learning trading signal using Alpaca market data and paper-trading-safe infrastructure.

## Current Work Split

Charles owns:

- Alpaca daily OHLCV data retrieval
- Feature engineering
- PCA feature compression
- Machine-learning model
- Historical signal generation
- Signal/PCA charts
- Repo organization and README

Patrick owns:

- Long-only backtest
- Performance metrics
- Paper-trading order execution demo
- Alpaca dashboard screenshots/logs
- Final video recording/editing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with Alpaca credentials. Use paper-trading credentials and paper-only accounts for the demo.

## Generate Charles's Signal Output

```bash
python -m hw3_signal_pipeline.run_pipeline --ticker AAPL
```

Optional:

```bash
python -m hw3_signal_pipeline.run_pipeline --ticker MSFT --years 5 --threshold 0.6 --output-dir outputs/msft
```

The pipeline saves:

- `outputs/<ticker>/signal_dataset.csv`
- `outputs/<ticker>/pca_variance.png`
- `outputs/<ticker>/signal_probability.png`
- `outputs/<ticker>/feature_correlation.png`
- `models/<ticker>_signal_model.joblib`

## Interface For Patrick

Patrick can backtest from `signal_dataset.csv`. The most important columns are:

- `close`: close price
- `forward_return`: next-day return used to define the target
- `target`: `1` when next-day return is positive, else `0`
- `model_prob`: model probability that next-day return is positive
- `signal`: `1` for long when `model_prob > 0.6`, else `0` for flat
- `sample`: `train` or `test`; backtest results should emphasize the `test` rows
- `PC1`, `PC2`, etc.: PCA features used by the model

The signal is long-only and never short.

## Latest Signal

After running the training pipeline once, get a fresh latest signal with:

```bash
python -m hw3_signal_pipeline.latest_signal --ticker AAPL --artifact models/AAPL_signal_model.joblib
```

This script only prints the latest signal. It does not submit orders. The paper-trading execution script should remain paper-only.

## Patrick's Backtest

After generating the signal dataset, backtest the ML signal against Buy & Hold:

```bash
python -m hw3_signal_pipeline.backtest --ticker AAPL
```

The backtest reads `outputs/aapl/signal_dataset.csv` and runs on the
out-of-sample `test` rows by default (`--sample all` includes train rows).
Constraints: $100,000 initial capital, long-only, no leverage, no short
selling, and signals formed at today's close execute on the next bar. It
tracks portfolio value, daily returns, every round-trip trade, and P&L, and
computes Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown, and
Win Rate for both strategies. It saves:

- `outputs/aapl/equity_curves.png`
- `outputs/aapl/drawdowns.png`
- `outputs/aapl/backtest_metrics.csv`
- `outputs/aapl/trades.csv`

## Paper Trading Demo

```bash
python -m hw3_signal_pipeline.paper_trader --ticker AAPL --artifact models/AAPL_signal_model.joblib
```

The script fetches the latest data, rebuilds the features, applies the saved
scaler + PCA + model, logs the signal, and routes the decision to Alpaca's
**paper** endpoint (the trading client is hard-coded to `paper=True`):

- signal = LONG and no open position: submit a market BUY (default $10,000
  notional, configurable with `--notional`)
- signal = LONG and already long: hold, no order
- signal = FLAT and already long: close the position (market SELL)
- signal = FLAT and no position: nothing to do

Use `--dry-run` to log the decision without submitting an order. Run it
during market hours to see the fill immediately in the Alpaca paper
dashboard; outside market hours the order stays queued as a day order.

**Paper trading only. No real money is ever used.**

## Submitted Charts

`outputs/` and `models/` are gitignored working directories; `charts/` holds
the committed artifacts from the submitted AAPL run: PCA variance, signal
probability, feature correlation, equity curves, drawdowns, and the backtest
metrics table.

## Tests

```bash
pip install pytest
pytest
```

## Video

Video link: _add the unlisted YouTube link here before submission_.

The 3-6 minute video shows the code running, the charts (equity curve,
drawdown, PCA variance), the backtest results, the Alpaca paper trading
dashboard, and a paper trade being executed -- with the statement:
"This is paper trading only -- no real money is used."
