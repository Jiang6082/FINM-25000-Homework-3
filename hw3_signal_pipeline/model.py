from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from hw3_signal_pipeline.features import FEATURE_COLUMNS


@dataclass
class SignalModelBundle:
    ticker: str
    feature_columns: list[str]
    scaler: StandardScaler
    pca: PCA
    model: RandomForestClassifier
    threshold: float
    explained_variance_target: float


def _component_count_for_variance(pca: PCA, target: float) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, target) + 1)


def train_signal_model(
    dataset: pd.DataFrame,
    ticker: str,
    threshold: float = 0.6,
    explained_variance_target: float = 0.80,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, SignalModelBundle]:
    """Fit scaler, PCA, and Random Forest using a chronological train/test split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    data = dataset.copy()
    split_index = int(len(data) * train_fraction)
    if split_index < 100 or len(data) - split_index < 20:
        raise ValueError("Not enough rows for a stable train/test split.")

    X = data[FEATURE_COLUMNS]
    y = data["target"].astype(int)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    full_pca = PCA()
    full_pca.fit(X_train_scaled)
    n_components = _component_count_for_variance(full_pca, explained_variance_target)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_pca, y_train)

    X_all_pca = pca.transform(scaler.transform(X))
    probabilities = model.predict_proba(X_all_pca)[:, 1]

    for idx in range(n_components):
        data[f"PC{idx + 1}"] = X_all_pca[:, idx]
    data["model_prob"] = probabilities
    data["signal"] = (data["model_prob"] > threshold).astype(int)
    data["sample"] = np.where(np.arange(len(data)) < split_index, "train", "test")

    bundle = SignalModelBundle(
        ticker=ticker.upper(),
        feature_columns=FEATURE_COLUMNS.copy(),
        scaler=scaler,
        pca=pca,
        model=model,
        threshold=threshold,
        explained_variance_target=explained_variance_target,
    )
    return data, bundle


def predict_latest_signal(dataset: pd.DataFrame, bundle: SignalModelBundle) -> dict:
    """Return the latest model probability and long/flat signal."""
    latest = dataset.dropna(subset=bundle.feature_columns).iloc[-1]
    X = latest[bundle.feature_columns].to_frame().T
    components = bundle.pca.transform(bundle.scaler.transform(X))
    probability = float(bundle.model.predict_proba(components)[0, 1])
    signal = int(probability > bundle.threshold)
    return {
        "date": str(latest.name),
        "ticker": bundle.ticker,
        "model_prob": probability,
        "signal": signal,
        "action": "LONG" if signal == 1 else "FLAT",
    }
