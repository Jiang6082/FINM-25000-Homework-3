import numpy as np
import pandas as pd

from hw3_signal_pipeline.features import FEATURE_COLUMNS, build_supervised_dataset
from hw3_signal_pipeline.model import train_signal_model


def synthetic_prices(rows=320):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=rows, freq="B")
    returns = rng.normal(0.0008, 0.018, size=rows)
    close = 100 * np.exp(np.cumsum(returns))
    spread = rng.uniform(0.002, 0.025, size=rows)
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = close * (1 + rng.normal(0, 0.004, size=rows))
    volume = rng.integers(1_000_000, 5_000_000, size=rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_features_and_model_produce_signal_columns():
    dataset = build_supervised_dataset(synthetic_prices())
    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert {"forward_return", "target"}.issubset(dataset.columns)

    signal_data, bundle = train_signal_model(dataset, ticker="AAPL")
    assert bundle.pca.explained_variance_ratio_.sum() >= 0.8
    assert {"model_prob", "signal", "sample", "PC1"}.issubset(signal_data.columns)
    assert set(signal_data["signal"].unique()).issubset({0, 1})
