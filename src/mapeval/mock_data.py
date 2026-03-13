"""Generate mock market data for testing without network access."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd


def generate_mock_klines(
    symbols: List[str],
    interval: str = "1m",
    lookback: int = 500,
    base_prices: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate realistic mock OHLCV data for testing.

    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        interval: Kline interval (1m, 5m, etc.)
        lookback: Number of candles to generate
        base_prices: Starting prices for each symbol

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    if base_prices is None:
        base_prices = {
            "BTCUSDT": 95000.0,
            "ETHUSDT": 3200.0,
            "BNBUSDT": 650.0,
            "SOLUSDT": 180.0,
            "XRPUSDT": 2.5,
            "DOGEUSDT": 0.35,
            "ADAUSDT": 0.95,
        }

    # Interval to minutes
    interval_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
    }.get(interval, 1)

    rows = []
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(minutes=interval_minutes * lookback)

    for symbol in symbols:
        price = base_prices.get(symbol, 100.0)
        volatility = price * 0.001  # 0.1% per candle

        for i in range(lookback):
            ts = start_time + timedelta(minutes=interval_minutes * i)

            # Random walk with slight trend
            drift = random.gauss(0, 1) * volatility
            price = max(price * 0.9, price + drift)  # Prevent negative

            # Generate OHLC
            open_price = price
            high_price = price * (1 + abs(random.gauss(0, 0.0005)))
            low_price = price * (1 - abs(random.gauss(0, 0.0005)))
            close_price = price + random.gauss(0, volatility * 0.5)
            close_price = max(low_price, min(high_price, close_price))

            # Update price for next candle
            price = close_price

            # Volume (higher for BTC/ETH)
            base_volume = {"BTCUSDT": 1000, "ETHUSDT": 5000}.get(symbol, 10000)
            volume = base_volume * random.uniform(0.5, 2.0)

            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
            })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


class MockMarketData:
    """Mock market data provider for testing without network."""

    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        lookback: int = 500,
    ):
        self.symbols = symbols
        self.interval = interval
        self.lookback = lookback

        # Generate initial history
        self._df = generate_mock_klines(symbols, interval, lookback)
        self._current_idx = lookback - 1
        self._prices = {}

        # Set initial prices from last row of each symbol
        for symbol in symbols:
            symbol_df = self._df[self._df["symbol"] == symbol]
            if not symbol_df.empty:
                self._prices[symbol] = symbol_df.iloc[-1]["close"]

    def fetch_latest_prices(self) -> Dict[str, float]:
        """Simulate fetching latest prices with small random movement."""
        for symbol in self.symbols:
            current = self._prices.get(symbol, 100.0)
            volatility = current * 0.0002  # 0.02% per tick
            self._prices[symbol] = max(1.0, current + random.gauss(0, volatility))
        return self._prices.copy()

    def latest_prices(self) -> Dict[str, float]:
        return self._prices.copy()

    def append_prices(self, prices: Dict[str, float], timestamp) -> None:
        """Append new prices to history."""
        self._prices.update(prices)

    def get_recent_window(self, window: int = 100) -> pd.DataFrame:
        """Return recent OHLCV data."""
        return self._df.tail(window * len(self.symbols)).copy()

    def refresh_funding_rates(self, throttle_seconds: int = 0) -> Dict[str, float]:
        """Return mock funding rates."""
        return {symbol: random.uniform(-0.001, 0.001) for symbol in self.symbols}

    def start_websocket(self) -> None:
        pass

    def stop_websocket(self) -> None:
        pass


if __name__ == "__main__":
    # Test the mock data generator
    symbols = ["BTCUSDT", "ETHUSDT"]
    df = generate_mock_klines(symbols, "1m", 100)
    print(df.head(20))
    print(f"\nGenerated {len(df)} rows for {len(symbols)} symbols")

    # Test MockMarketData
    market = MockMarketData(symbols, "1m", 100)
    print(f"\nInitial prices: {market.latest_prices()}")
    for _ in range(5):
        print(f"Tick: {market.fetch_latest_prices()}")
