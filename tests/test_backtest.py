import numpy as np
import pandas as pd

from hw3_signal_pipeline.backtest import (
    INITIAL_CAPITAL,
    METRIC_ORDER,
    backtest_signal_dataset,
    metrics_table,
    run_long_only_backtest,
)


def synthetic_signal_dataset(rows=200):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, rows))), index=dates)
    model_prob = pd.Series(rng.uniform(0.3, 0.9, rows), index=dates)
    split = int(rows * 0.8)
    return pd.DataFrame(
        {
            "close": close,
            "model_prob": model_prob,
            "signal": (model_prob > 0.6).astype(int),
            "sample": np.where(np.arange(rows) < split, "train", "test"),
        }
    )


def test_equity_starts_at_initial_capital_and_metrics_are_complete():
    data = synthetic_signal_dataset()
    buy_hold, ml_signal = backtest_signal_dataset(data, sample="test")

    for result in (buy_hold, ml_signal):
        assert result.equity.iloc[0] == INITIAL_CAPITAL
        assert set(METRIC_ORDER) == set(result.metrics)
        assert 0.0 <= result.metrics["win_rate"] <= 1.0
        assert result.metrics["max_drawdown"] <= 0.0
        assert np.isclose(result.total_pnl, result.equity.iloc[-1] - INITIAL_CAPITAL)

    table = metrics_table([buy_hold, ml_signal])
    assert list(table.index) == ["Buy & Hold", "ML Signal"]
    assert list(table.columns) == METRIC_ORDER


def test_backtest_uses_only_the_requested_sample_rows():
    data = synthetic_signal_dataset()
    _, ml_signal = backtest_signal_dataset(data, sample="test")
    test_rows = data[data["sample"] == "test"]
    assert len(ml_signal.equity) == len(test_rows)
    assert ml_signal.equity.index[0] == test_rows.index[0]


def test_signals_execute_on_the_next_bar():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)
    position = pd.Series([1, 1, 1, 1], index=dates)

    result = run_long_only_backtest(close, position)
    # The first bar earns nothing: a signal formed at its close is applied to
    # the next bar, so there is no look-ahead.
    assert result.daily_returns.iloc[0] == 0.0
    assert np.isclose(result.daily_returns.iloc[1], 0.10)
    assert np.isclose(result.equity.iloc[-1], INITIAL_CAPITAL * 1.1**3)


def test_positions_are_clipped_to_long_only_no_leverage():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    close = pd.Series(np.linspace(100, 130, 50), index=dates)

    short = run_long_only_backtest(close, pd.Series(-1, index=dates))
    assert (short.daily_returns == 0).all()

    levered = run_long_only_backtest(close, pd.Series(2, index=dates))
    unlevered = run_long_only_backtest(close, pd.Series(1, index=dates))
    assert np.allclose(levered.daily_returns, unlevered.daily_returns)


def test_trade_log_is_consistent_with_the_equity_curve():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.Series([100.0, 100.0, 110.0, 110.0, 99.0, 99.0], index=dates)
    # Signal turns on at bar 1, so the position is held over bars 2-3:
    # entered at bar 1's close (100), exited at bar 3's close (110).
    position = pd.Series([0, 1, 1, 0, 0, 0], index=dates)

    result = run_long_only_backtest(close, position)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry"] == dates[1]
    assert trade["exit"] == dates[3]
    assert np.isclose(trade["return"], 0.10)
    assert np.isclose(trade["pnl"], INITIAL_CAPITAL * 0.10)
    # The single trade's P&L is exactly what the equity curve realizes.
    assert np.isclose(result.equity.iloc[-1], INITIAL_CAPITAL + trade["pnl"])
