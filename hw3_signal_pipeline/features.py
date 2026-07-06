from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_return",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_20",
    "rolling_std_20",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rsi_14",
    "stochastic_k_14",
    "williams_r_14",
    "bollinger_pct_b",
    "bollinger_bandwidth",
    "atr_14",
    "obv",
    "cmf_20",
]


def add_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators across trend, momentum, volatility, and volume."""
    df = price_data.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["log_return"] = np.log(close / close.shift(1))
    df["rolling_mean_10"] = df["log_return"].rolling(10).mean()
    df["rolling_std_10"] = df["log_return"].rolling(10).std()
    df["rolling_mean_20"] = df["log_return"].rolling(20).mean()
    df["rolling_std_20"] = df["log_return"].rolling(20).std()

    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    df["stochastic_k_14"] = 100 * (close - low_14) / range_14
    df["williams_r_14"] = -100 * (high_14 - close) / range_14

    bollinger_mid = close.rolling(20).mean()
    bollinger_std = close.rolling(20).std()
    bollinger_upper = bollinger_mid + 2 * bollinger_std
    bollinger_lower = bollinger_mid - 2 * bollinger_std
    bollinger_width = (bollinger_upper - bollinger_lower).replace(0, np.nan)
    df["bollinger_pct_b"] = (close - bollinger_lower) / bollinger_width
    df["bollinger_bandwidth"] = bollinger_width / bollinger_mid.replace(0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()

    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(
        0, np.nan
    )
    money_flow_volume = money_flow_multiplier * volume
    df["cmf_20"] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum()

    return df


def build_supervised_dataset(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create features plus a binary next-day-return target."""
    df = add_features(price_data)
    df["forward_return"] = df["close"].pct_change().shift(-1)
    df["target"] = (df["forward_return"] > 0).astype("Int64")
    df.loc[df["forward_return"].isna(), "target"] = pd.NA
    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
