from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw3_signal_pipeline.features import FEATURE_COLUMNS
from hw3_signal_pipeline.model import SignalModelBundle


def save_pca_variance_chart(bundle: SignalModelBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    explained = bundle.pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, explained, label="Individual variance")
    ax.plot(x, cumulative, marker="o", color="black", label="Cumulative variance")
    ax.axhline(bundle.explained_variance_target, color="red", linestyle="--")
    ax.set_title(f"{bundle.ticker} PCA Explained Variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "pca_variance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_signal_probability_chart(data: pd.DataFrame, bundle: SignalModelBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data.index, data["model_prob"], label="Probability next-day return > 0")
    ax.axhline(bundle.threshold, color="red", linestyle="--", label="Long threshold")
    ax.set_title(f"{bundle.ticker} ML Signal Probability")
    ax.set_xlabel("Date")
    ax.set_ylabel("Model Probability")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "signal_probability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_correlation_chart(data: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    corr = data[FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix")
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_yticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = output_dir / "feature_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
