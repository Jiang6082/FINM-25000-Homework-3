from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
from alpaca.data.enums import DataFeed

from hw3_signal_pipeline.data import _load_keys, fetch_daily_bars
from hw3_signal_pipeline.features import add_features
from hw3_signal_pipeline.model import predict_latest_signal


log = logging.getLogger("paper_trader")


def decide_action(action: str, has_position: bool) -> str:
    """Map (LONG/FLAT signal, current position) to an order action.

    Buy when LONG with no position, close the position when FLAT, otherwise
    do nothing. Pure and testable.
    """
    if action == "LONG":
        return "hold" if has_position else "buy"
    return "sell" if has_position else "none"


def make_trading_client():
    """Construct the Alpaca trading client. Always the PAPER endpoint."""
    from alpaca.trading.client import TradingClient

    api_key, secret_key = _load_keys()
    return TradingClient(api_key, secret_key, paper=True)


def get_open_position(client, ticker: str):
    """Return the open position for ``ticker`` or None if there is none."""
    try:
        return client.get_open_position(ticker)
    except Exception:
        return None


def submit_buy(client, ticker: str, notional: float):
    """Submit a notional market BUY (day order) on the paper account."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    request = MarketOrderRequest(
        symbol=ticker,
        notional=round(notional, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(request)


def run_once(ticker: str, artifact: str, years: int, feed: DataFeed, notional: float, dry_run: bool) -> dict:
    """Fetch data, score the latest bar with the trained bundle, route one paper trade."""
    log.info("=" * 68)
    log.info("Alpaca PAPER trading demo -- no real money is used.")
    log.info("=" * 68)

    bundle = joblib.load(artifact)
    if bundle.ticker != ticker:
        log.warning("Artifact was trained on %s but trading %s.", bundle.ticker, ticker)

    log.info("Fetching latest daily bars for %s from Alpaca...", ticker)
    prices = fetch_daily_bars(ticker, years=years, feed=feed)
    log.info(
        "Loaded %d daily bars (%s -> %s).",
        len(prices),
        prices.index[0].date(),
        prices.index[-1].date(),
    )

    # add_features keeps the newest bar; build_supervised_dataset would drop it
    # because its next-day target is still unknown -- exactly the bar we trade.
    log.info("Computing features and applying the saved scaler + PCA + model...")
    latest = predict_latest_signal(add_features(prices), bundle)
    log.info(
        "SIGNAL %s date=%s model_prob=%.4f threshold=%.2f -> %s",
        ticker,
        latest["date"],
        latest["model_prob"],
        bundle.threshold,
        latest["action"],
    )

    client = make_trading_client()
    account = client.get_account()
    log.info(
        "Connected to PAPER account: status=%s buying_power=$%s",
        account.status,
        account.buying_power,
    )

    position = get_open_position(client, ticker)
    if position is not None:
        log.info("Current position: %s shares of %s.", position.qty, ticker)
    else:
        log.info("Current position: none.")

    order_action = decide_action(latest["action"], position is not None)

    if order_action == "buy":
        if dry_run:
            log.info("DRY RUN: would submit market BUY $%.2f of %s.", notional, ticker)
        else:
            order = submit_buy(client, ticker, notional)
            log.info(
                "ORDER submitted: id=%s side=buy notional=$%.2f status=%s",
                order.id,
                notional,
                order.status,
            )
    elif order_action == "sell":
        if dry_run:
            log.info("DRY RUN: would close the %s position (market SELL).", ticker)
        else:
            order = client.close_position(ticker)
            log.info("ORDER submitted: id=%s side=sell status=%s", order.id, order.status)
    elif order_action == "hold":
        log.info("Signal is LONG and the position is already open -- holding, no order.")
    else:
        log.info("Signal is FLAT and there is no position -- nothing to do.")

    log.info("Done. Review the trade in the Alpaca paper dashboard.")
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route the latest ML signal to Alpaca paper trading.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--artifact", required=True, help="Path to *_signal_model.joblib")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    parser.add_argument("--notional", type=float, default=10_000.0, help="Dollar size of a new BUY.")
    parser.add_argument("--dry-run", action="store_true", help="Log the decision, submit nothing.")
    parser.add_argument("--log-file", default=None, help="Also write the logs to this file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    feed = DataFeed.IEX if args.feed == "iex" else DataFeed.SIP
    run_once(args.ticker.upper().strip(), args.artifact, args.years, feed, args.notional, args.dry_run)


if __name__ == "__main__":
    main()
