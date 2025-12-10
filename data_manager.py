"""Market data utilities for real-time leveraged futures trading.

This module provides both legacy helpers for synthetic equity data generation
and new abstractions tailored to real-time cryptocurrency futures trading. The
`RealTimeMarketData` class bootstraps an intraday price history from Binance
and keeps it updated with the latest ticker snapshots so downstream components
always have a clean pandas DataFrame to work with.
"""

from __future__ import annotations

import math
from typing import Dict, Final, Iterable, List, Optional, Sequence

import pandas as pd

from binance_data_source import (
    DEFAULT_BASE,
    FALLBACK_BASES,
    check_api,
    fetch_klines,
    get_ticker_price,
    get_futures_premium_index,
)
from binance_ws import BinanceSpotWS


_START_DATE: Final = "2015-01-01"
_END_DATE: Final = "2024-12-31"
_INTERVAL_TO_FREQ: Dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "B",
}


def _klines_to_df(klines: List[List]) -> pd.DataFrame:
    """Convert Binance kline arrays to a DataFrame with timestamp and close price."""
    records = []
    for kline in klines:
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
        ts = int(kline[0])
        close = float(kline[4])
        records.append({"Date": pd.to_datetime(ts, unit="ms"), "Close": close})
    return pd.DataFrame(records)


def _interval_to_freq(interval: str) -> str:
    return _INTERVAL_TO_FREQ.get(interval, "T")


def _fetch_symbol_history(symbol: str, interval: str, lookback: int, base_url: str) -> pd.DataFrame:
    klines = fetch_klines(symbol=symbol, interval=interval, limit=lookback, base_url=base_url)
    df = _klines_to_df(klines)
    if df.empty:
        raise ValueError("No kline data returned.")
    return df


class RealTimeMarketData:
    """Maintain an up-to-date intraday price history for multiple symbols."""

    def __init__(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
        lookback: int = 500,
        base_url: str = DEFAULT_BASE,
    ) -> None:
        self.symbols = list(symbols)
        if not self.symbols:
            raise ValueError("At least one symbol must be provided.")
        self.interval = interval
        self.lookback = lookback
        self.base_url = base_url
        self._columns = [f"{symbol}_Close" for symbol in self.symbols]
        self.price_history = self._bootstrap_history()
        self._ws: Optional[BinanceSpotWS] = None
        self._funding_cache: Dict[str, float] = {}
        self._last_funding_fetch_ts: Optional[pd.Timestamp] = None

    def _bootstrap_history(self) -> pd.DataFrame:
        frames = []
        for symbol in self.symbols:
            history = _fetch_symbol_history(symbol, self.interval, self.lookback, self.base_url)
            history = history.rename(columns={"Close": f"{symbol}_Close"})
            history = history.set_index("Date")
            if getattr(history.index, "tz", None) is not None:
                history.index = history.index.tz_convert(None)
            frames.append(history)
        merged = pd.concat(frames, axis=1).sort_index()
        if getattr(merged.index, "tz", None) is not None:
            merged.index = merged.index.tz_convert(None)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.ffill()
        merged = merged.tail(self.lookback)
        merged.index.name = "Date"
        # Ensure consistent column order even if concat reorders.
        merged = merged.reindex(columns=self._columns)
        return merged

    def refresh_history(self) -> None:
        """Force a history refresh from the remote API."""
        self.price_history = self._bootstrap_history()

    def fetch_latest_prices(self) -> Dict[str, float]:
        """Retrieve the most recent ticker price for all configured symbols."""
        # Prefer websocket snapshot if running and populated
        if self._ws is not None:
            ws_prices = self._ws.get_latest_prices()
            if all(sym in ws_prices for sym in self.symbols):
                return {sym: float(ws_prices[sym]) for sym in self.symbols}
        try:
            payload = get_ticker_price(symbols=self.symbols, base_url=self.base_url)
            if isinstance(payload, dict):
                payload = [payload]
            prices: Dict[str, float] = {}
            for item in payload:
                symbol = str(item.get("symbol"))
                if symbol in self.symbols and "price" in item:
                    prices[symbol] = float(item["price"])
            if len(prices) != len(self.symbols):
                raise ValueError("Incomplete ticker payload received.")
            return prices
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to fetch latest prices: {exc}") from exc

    def start_websocket(self) -> None:
        if self._ws is None:
            self._ws = BinanceSpotWS(self.symbols)
            self._ws.start()

    def stop_websocket(self) -> None:
        if self._ws is not None:
            self._ws.stop()
            self._ws = None

    def refresh_funding_rates(self, throttle_seconds: int = 60) -> Dict[str, float]:
        now = pd.Timestamp.utcnow().floor("s")
        if self._last_funding_fetch_ts is not None:
            delta = (now - self._last_funding_fetch_ts).total_seconds()
            if delta < throttle_seconds:
                return dict(self._funding_cache)
        try:
            data = get_futures_premium_index()
            cache: Dict[str, float] = {}
            if isinstance(data, list):
                for item in data:
                    sym = str(item.get("symbol", "")).upper()
                    if sym in self.symbols and "lastFundingRate" in item:
                        try:
                            cache[sym] = float(item["lastFundingRate"])  # per 8h
                        except Exception:
                            continue
            elif isinstance(data, dict):
                sym = str(data.get("symbol", "")).upper()
                if sym in self.symbols and "lastFundingRate" in data:
                    cache[sym] = float(data["lastFundingRate"])  # per 8h
            if cache:
                self._funding_cache.update(cache)
                self._last_funding_fetch_ts = now
        except Exception:
            pass
        return dict(self._funding_cache)

    def append_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp | None = None) -> None:
        """Append the newest price snapshot to the internal history buffer."""
        timestamp = pd.to_datetime(timestamp or pd.Timestamp.utcnow().floor("s"))
        if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        row = {f"{symbol}_Close": float(prices.get(symbol)) for symbol in self.symbols}
        df = pd.DataFrame([row], index=[timestamp])
        df.index.name = "Date"
        history = pd.concat([self.price_history, df])
        if getattr(history.index, "tz", None) is not None:
            history.index = history.index.tz_convert(None)
        history = history[~history.index.duplicated(keep="last")].sort_index()
        history = history.ffill().tail(self.lookback)
        self.price_history = history

    def get_recent_window(self, rows: int = 120) -> pd.DataFrame:
        """Return the most recent slice of price history as a DataFrame with a Date column."""
        window = self.price_history.tail(rows).copy()
        window = window.reset_index()
        return window

    def latest_prices(self) -> Dict[str, float]:
        last_row = self.price_history.iloc[-1]
        return {
            symbol: float(last_row[f"{symbol}_Close"])
            for symbol in self.symbols
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full buffered history with Date column included."""
        return self.price_history.reset_index()


def _attempt_binance_load(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Try to fetch daily klines for `symbol` from Binance and return a DataFrame."""
    api_status = check_api(DEFAULT_BASE)
    if not api_status.get("reachable"):
        raise ConnectionError(f"Binance unreachable: {api_status.get('error')}")

    klines = fetch_klines(symbol=symbol, interval=interval, limit=1000)
    df = _klines_to_df(klines)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
    return df
