from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIAL_CAPITAL = 100_000.0
TRADING_DAYS = 252

REQUIRED_COLUMNS = ["close", "model_prob", "signal", "sample"]
METRIC_ORDER = [
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
]


@dataclass
class BacktestResult:
    """Everything tracked for one strategy: value, returns, trades, and scores."""

    name: str
    equity: pd.Series
    daily_returns: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]

    @property
    def total_pnl(self) -> float:
        return float(self.equity.iloc[-1] - self.equity.iloc[0])


def run_long_only_backtest(
    close: pd.Series,
    position: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    name: str = "Strategy",
) -> BacktestResult:
    """Backtest a 0/1 long-only position series with next-day execution.

    A signal formed at today's close is applied to the next bar, so there is
    no look-ahead. Positions are clipped to 0/1: long-only, no leverage, no
    short selling.
    """
    held = position.reindex(close.index).fillna(0).clip(0, 1).shift(1).fillna(0)
    market_return = close.pct_change().fillna(0)
    daily_returns = market_return * held
    equity = initial_capital * (1 + daily_returns).cumprod()
    trades = _trade_log(close, held, equity)
    return BacktestResult(
        name=name,
        equity=equity,
        daily_returns=daily_returns,
        drawdown=compute_drawdown(equity),
        trades=trades,
        metrics=compute_metrics(equity, daily_returns, trades),
    )


def backtest_signal_dataset(
    data: pd.DataFrame,
    sample: str = "test",
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[BacktestResult, BacktestResult]:
    """Backtest Buy & Hold and the ML signal on the requested sample rows.

    ``sample`` is ``"test"`` (default, the out-of-sample rows the model never
    trained on), ``"train"``, or ``"all"``.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Signal dataset is missing columns: {missing}")

    rows = data if sample == "all" else data[data["sample"] == sample]
    if len(rows) < 20:
        raise ValueError(f"Not enough '{sample}' rows to backtest ({len(rows)}).")

    close = rows["close"]
    buy_hold = run_long_only_backtest(
        close, pd.Series(1, index=close.index), initial_capital, name="Buy & Hold"
    )
    ml_signal = run_long_only_backtest(
        close, rows["signal"], initial_capital, name="ML Signal"
    )
    return buy_hold, ml_signal


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Peak-to-trough decline of the equity curve, as a fraction <= 0."""
    return equity / equity.cummax() - 1


def compute_metrics(
    equity: pd.Series, daily_returns: pd.Series, trades: pd.DataFrame
) -> dict[str, float]:
    """Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Win Rate."""
    elapsed_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / TRADING_DAYS)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / elapsed_years) - 1)
    volatility = float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    annualized_return = float(daily_returns.mean() * TRADING_DAYS)
    downside_returns = daily_returns[daily_returns < 0]
    downside = (
        float(downside_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(downside_returns) else 0.0
    )
    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": annualized_return / volatility if volatility > 0 else 0.0,
        "sortino": annualized_return / downside if downside > 0 else 0.0,
        "max_drawdown": float(compute_drawdown(equity).min()),
        "win_rate": float((trades["return"] > 0).mean()) if len(trades) else 0.0,
    }
    return {key: value if np.isfinite(value) else 0.0 for key, value in metrics.items()}


def metrics_table(results: list[BacktestResult]) -> pd.DataFrame:
    """One row per strategy, one column per required performance metric."""
    return pd.DataFrame(
        {result.name: [result.metrics[key] for key in METRIC_ORDER] for result in results},
        index=METRIC_ORDER,
    ).T


def _trade_log(close: pd.Series, held: pd.Series, equity: pd.Series) -> pd.DataFrame:
    """Round-trip trades, exactly consistent with the equity curve.

    A trade is entered at the close of the bar where the signal turned on and
    exited at the close of the last bar it was held, so each trade's return
    compounds to the same P&L the equity curve realizes.
    """
    changes = held.diff().fillna(held).to_numpy()
    entry_marks = np.flatnonzero(changes > 0)  # first held bar of each trade
    exit_marks = list(np.flatnonzero(changes < 0))  # first flat bar after each trade
    if len(exit_marks) < len(entry_marks):  # still holding at the end
        exit_marks.append(len(close))

    rows = []
    for first_held, first_flat in zip(entry_marks, exit_marks):
        entry_pos = first_held - 1  # the signal bar whose close we entered at
        exit_pos = min(first_flat, len(close)) - 1  # the last held bar
        trade_return = float(close.iloc[exit_pos] / close.iloc[entry_pos] - 1)
        deployed = float(equity.iloc[entry_pos])  # capital at the moment of entry
        rows.append(
            {
                "entry": close.index[entry_pos],
                "exit": close.index[exit_pos],
                "entry_price": float(close.iloc[entry_pos]),
                "exit_price": float(close.iloc[exit_pos]),
                "return": trade_return,
                "pnl": deployed * trade_return,
            }
        )
    return pd.DataFrame(rows, columns=["entry", "exit", "entry_price", "exit_price", "return", "pnl"])


def save_equity_curve_chart(
    results: list[BacktestResult], ticker: str, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for result in results:
        ax.plot(result.equity.index, result.equity, label=result.name)
    ax.set_title(f"{ticker} Portfolio Value ($100,000 start, long-only)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "equity_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_drawdown_chart(results: list[BacktestResult], ticker: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for result in results:
        ax.plot(result.drawdown.index, result.drawdown * 100, label=result.name)
    ax.set_title(f"{ticker} Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "drawdowns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_signal_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV written by run_pipeline with a proper DatetimeIndex."""
    if not path.exists():
        raise SystemExit(
            f"Error: {path} not found. Run "
            "'python -m hw3_signal_pipeline.run_pipeline --ticker <TICKER>' first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the ML signal vs Buy & Hold.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker used by run_pipeline.")
    parser.add_argument("--dataset", default=None, help="Path to signal_dataset.csv.")
    parser.add_argument(
        "--sample",
        default="test",
        choices=["test", "train", "all"],
        help="Rows to backtest; 'test' keeps the evaluation out of sample.",
    )
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--output-dir", default=None, help="Directory for charts/CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    dataset_path = Path(args.dataset or f"outputs/{ticker.lower()}/signal_dataset.csv")
    output_dir = Path(args.output_dir or dataset_path.parent)

    data = load_signal_dataset(dataset_path)
    buy_hold, ml_signal = backtest_signal_dataset(data, args.sample, args.initial_capital)
    results = [buy_hold, ml_signal]

    table = metrics_table(results)
    table.to_csv(output_dir / "backtest_metrics.csv")
    ml_signal.trades.to_csv(output_dir / "trades.csv", index=False)
    chart_paths = [
        save_equity_curve_chart(results, ticker, output_dir),
        save_drawdown_chart(results, ticker, output_dir),
    ]

    window = ml_signal.equity.index
    print(f"Ticker: {ticker}")
    print(f"Backtest sample: {args.sample} ({len(window)} rows, {window[0].date()} -> {window[-1].date()})")
    print(f"Initial capital: ${args.initial_capital:,.0f} (long-only, no leverage, no shorts)")
    print()
    percent_rows = ["total_return", "cagr", "volatility", "max_drawdown", "win_rate"]
    display = table.copy()
    for column in display.columns:
        if column in percent_rows:
            display[column] = (display[column] * 100).map("{:.1f}%".format)
        else:
            display[column] = display[column].map("{:.2f}".format)
    print(display.to_string())
    print()
    print(f"ML Signal trades: {len(ml_signal.trades)}")
    print(f"ML Signal total P&L: ${ml_signal.total_pnl:,.2f}")
    print(f"Buy & Hold total P&L: ${buy_hold.total_pnl:,.2f}")
    print(f"Saved metrics: {output_dir / 'backtest_metrics.csv'}")
    print(f"Saved trades: {output_dir / 'trades.csv'}")
    for path in chart_paths:
        print(f"Saved chart: {path}")


if __name__ == "__main__":
    main()
