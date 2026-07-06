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
