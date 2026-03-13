"""Tests for Binance Futures symbol rule normalization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from mapeval.binance_futures_client import BinanceFuturesClient


def _client_with_exchange_info() -> BinanceFuturesClient:
    client = BinanceFuturesClient(api_key="a" * 24, api_secret="b" * 24, testnet=True)
    client._exchange_info_cache = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
                    {"filterType": "PRICE_FILTER", "minPrice": "0.10", "maxPrice": "1000000", "tickSize": "0.10"},
                    {"filterType": "MIN_NOTIONAL", "notional": "100"},
                ],
            }
        ]
    }
    return client


class TestBinanceOrderNormalization:
    def test_rounds_quantity_and_price_down_to_exchange_steps(self):
        client = _client_with_exchange_info()

        normalized = client.normalize_order("BTCUSDT", quantity=0.0059, price=63250.187)

        assert normalized["quantity"] == pytest.approx(0.005)
        assert normalized["price"] == pytest.approx(63250.1)

    def test_rejects_quantity_below_min_qty(self):
        client = _client_with_exchange_info()

        with pytest.raises(ValueError):
            client.normalize_order("BTCUSDT", quantity=0.0004, price=63250.0)

    def test_rejects_notional_below_min_notional(self):
        client = _client_with_exchange_info()

        with pytest.raises(ValueError):
            client.normalize_order("BTCUSDT", quantity=0.001, price=50.0)
