from __future__ import annotations

import argparse

import joblib
from alpaca.data.enums import DataFeed

from hw3_signal_pipeline.data import fetch_daily_bars
from hw3_signal_pipeline.features import build_supervised_dataset
from hw3_signal_pipeline.model import predict_latest_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the latest trained ML signal.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--artifact", required=True, help="Path to *_signal_model.joblib")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feed = DataFeed.IEX if args.feed == "iex" else DataFeed.SIP
    bundle = joblib.load(args.artifact)
    try:
        prices = fetch_daily_bars(args.ticker, years=args.years, feed=feed)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    dataset = build_supervised_dataset(prices)
    latest = predict_latest_signal(dataset, bundle)

    print(f"Ticker: {latest['ticker']}")
    print(f"Date: {latest['date']}")
    print(f"Model probability: {latest['model_prob']:.4f}")
    print(f"Signal: {latest['signal']} ({latest['action']})")


if __name__ == "__main__":
    main()
