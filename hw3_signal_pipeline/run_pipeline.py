from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from alpaca.data.enums import DataFeed

from hw3_signal_pipeline.data import DEFAULT_TICKERS, fetch_daily_bars
from hw3_signal_pipeline.features import build_supervised_dataset
from hw3_signal_pipeline.model import train_signal_model
from hw3_signal_pipeline.visuals import (
    save_feature_correlation_chart,
    save_pca_variance_chart,
    save_signal_probability_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PCA + ML trading signals.")
    parser.add_argument("--ticker", default="AAPL", help=f"Ticker, e.g. {DEFAULT_TICKERS}")
    parser.add_argument("--years", type=int, default=5, help="Years of daily Alpaca data.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Long threshold.")
    parser.add_argument("--variance", type=float, default=0.8, help="PCA variance target.")
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"], help="Alpaca data feed.")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/charts.")
    parser.add_argument("--model-dir", default="models", help="Directory for model artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    feed = DataFeed.IEX if args.feed == "iex" else DataFeed.SIP
    output_dir = Path(args.output_dir or f"outputs/{ticker.lower()}")
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        prices = fetch_daily_bars(ticker=ticker, years=args.years, feed=feed)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    dataset = build_supervised_dataset(prices)
    signal_data, bundle = train_signal_model(
        dataset,
        ticker=ticker,
        threshold=args.threshold,
        explained_variance_target=args.variance,
    )

    csv_path = output_dir / "signal_dataset.csv"
    artifact_path = model_dir / f"{ticker}_signal_model.joblib"
    signal_data.to_csv(csv_path)
    joblib.dump(bundle, artifact_path)

    chart_paths = [
        save_pca_variance_chart(bundle, output_dir),
        save_signal_probability_chart(signal_data, bundle, output_dir),
        save_feature_correlation_chart(signal_data, output_dir),
    ]

    latest = signal_data.iloc[-1]
    print(f"Ticker: {ticker}")
    print(f"Rows after feature engineering: {len(signal_data)}")
    print(f"PCA components kept: {bundle.pca.n_components_}")
    print(f"Cumulative PCA variance: {bundle.pca.explained_variance_ratio_.sum():.4f}")
    print(f"Latest model probability: {latest['model_prob']:.4f}")
    print(f"Latest signal: {'LONG' if latest['signal'] == 1 else 'FLAT'}")
    print(f"Saved dataset: {csv_path}")
    print(f"Saved model artifact: {artifact_path}")
    for path in chart_paths:
        print(f"Saved chart: {path}")


if __name__ == "__main__":
    main()
