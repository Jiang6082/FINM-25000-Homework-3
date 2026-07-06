import os
from unittest.mock import MagicMock, patch

from hw3_signal_pipeline.paper_trader import decide_action, get_open_position, make_trading_client


def test_decision_matrix_matches_the_assignment():
    # Buy when LONG with no position; sell (close) when FLAT with one.
    assert decide_action("LONG", has_position=False) == "buy"
    assert decide_action("LONG", has_position=True) == "hold"
    assert decide_action("FLAT", has_position=True) == "sell"
    assert decide_action("FLAT", has_position=False) == "none"


@patch.dict(os.environ, {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"}, clear=True)
@patch("alpaca.trading.client.TradingClient")
def test_trading_client_is_always_paper(trading_client):
    make_trading_client()
    trading_client.assert_called_once_with("key", "secret", paper=True)


def test_missing_position_is_treated_as_none():
    client = MagicMock()
    client.get_open_position.side_effect = RuntimeError("position does not exist")
    assert get_open_position(client, "AAPL") is None
