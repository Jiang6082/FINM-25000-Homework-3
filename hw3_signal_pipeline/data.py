from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


DEFAULT_TICKERS = ("AAPL", "MSFT", "SPY", "QQQ", "NVDA")


def _load_keys() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."
        )
    return api_key, secret_key


def fetch_daily_bars(
    ticker: str,
    years: int = 5,
    feed: DataFeed = DataFeed.IEX,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV bars from Alpaca into a single-symbol DataFrame."""
    symbol = ticker.upper().strip()
    api_key, secret_key = _load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    end_dt = end or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * years + 10)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed=feed,
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError(f"No Alpaca bars returned for {symbol}.")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")

    bars = bars.sort_index()
    bars.index.name = "date"
    columns = ["open", "high", "low", "close", "volume"]
    return bars[columns].dropna()
