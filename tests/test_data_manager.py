"""Unit tests for data_manager.py."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import mapeval.data_manager as data_manager
from mapeval.data_manager import BacktestMarketData, RealTimeMarketData


def _build_realtime_market_data(symbols: list[str], history: pd.DataFrame) -> RealTimeMarketData:
    market = object.__new__(RealTimeMarketData)
    market.symbols = symbols
    market.interval = "1m"
    market.lookback = len(history)
    market.base_urls = ["https://example.com"]
    market.base_url = market.base_urls[0]
    market._columns = [f"{symbol}_Close" for symbol in symbols]
    market.price_history = history.copy()
    market.price_history.index.name = "Date"
    market._latest_prices_cache = market._extract_latest_prices(market.price_history)
    market._ws = None
    market._funding_cache = {}
    market._last_funding_fetch_ts = None
    market._consecutive_fetch_failures = 0
    market._offline_until = None
    market._offline_backoff_seconds = 60.0
    return market


class TestRealTimeMarketData:
    def test_append_prices_updates_existing_timestamp_in_place(self):
        index = pd.date_range("2024-01-01", periods=3, freq="min", name="Date")
        history = pd.DataFrame(
            {
                "BTCUSDT_Close": [100.0, 101.0, 102.0],
                "ETHUSDT_Close": [200.0, 201.0, 202.0],
            },
            index=index,
        )
        market = _build_realtime_market_data(["BTCUSDT", "ETHUSDT"], history)

        market.append_prices({"BTCUSDT": 150.0}, timestamp=index[-1])

        assert len(market.price_history) == 3
        assert market.price_history.iloc[-1]["BTCUSDT_Close"] == 150.0
        assert market.price_history.iloc[-1]["ETHUSDT_Close"] == 202.0
        assert market.latest_prices() == {"BTCUSDT": 150.0, "ETHUSDT": 202.0}

    def test_append_prices_trims_to_lookback_and_preserves_missing_prices(self):
        index = pd.date_range("2024-01-01", periods=3, freq="min", name="Date")
        history = pd.DataFrame(
            {
                "BTCUSDT_Close": [100.0, 101.0, 102.0],
                "ETHUSDT_Close": [200.0, 201.0, 202.0],
            },
            index=index,
        )
        market = _build_realtime_market_data(["BTCUSDT", "ETHUSDT"], history)

        market.append_prices({"BTCUSDT": 103.0}, timestamp=index[-1] + pd.Timedelta(minutes=1))

        assert len(market.price_history) == 3
        assert list(market.price_history.index) == list(index[1:]) + [index[-1] + pd.Timedelta(minutes=1)]
        assert market.price_history.iloc[-1]["BTCUSDT_Close"] == 103.0
        assert market.price_history.iloc[-1]["ETHUSDT_Close"] == 202.0
        assert market.latest_prices() == {"BTCUSDT": 103.0, "ETHUSDT": 202.0}

    def test_bootstrap_history_fetches_symbols_concurrently(self, monkeypatch):
        thread_ids: set[int] = set()

        def fake_fetch(symbol: str, interval: str, lookback: int, base_url):
            thread_ids.add(threading.get_ident())
            time.sleep(0.05)
            index = pd.date_range("2024-01-01", periods=lookback, freq="min")
            return pd.DataFrame({"Date": index, "Close": [float(len(symbol))] * lookback})

        monkeypatch.setattr(data_manager, "_fetch_symbol_history", fake_fetch)

        market = object.__new__(RealTimeMarketData)
        market.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        market.interval = "1m"
        market.lookback = 10
        market.base_urls = ["https://example.com"]
        market.base_url = market.base_urls[0]
        market._columns = [f"{symbol}_Close" for symbol in market.symbols]

        result = market._bootstrap_history()

        assert result.shape == (10, 4)
        assert list(result.columns) == market._columns
        assert len(thread_ids) >= 2


class TestBacktestMarketData:
    def test_append_prices_maintains_lookback_window(self):
        index = pd.date_range("2024-01-01", periods=6, freq="min", name="Date")
        history = pd.DataFrame(
            {
                "BTCUSDT_Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "ETHUSDT_Close": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
            },
            index=index,
        )
        market = BacktestMarketData(history, ["BTCUSDT", "ETHUSDT"], lookback=3)

        market.append_prices({})
        market.append_prices({})

        assert len(market.price_history) == 3
        assert list(market.price_history.index) == list(index[2:5])
        assert market.latest_prices() == {"BTCUSDT": 104.0, "ETHUSDT": 204.0}
