"""Market data utilities for real-time leveraged futures trading.

This module provides abstractions tailored to real-time cryptocurrency futures trading. The
`RealTimeMarketData` class bootstraps an intraday price history from Binance
and keeps it updated with the latest ticker snapshots so downstream components
always have a clean pandas DataFrame to work with.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Dict, Final, Iterable, List, Optional, Sequence

import pandas as pd

from mapeval.binance_data_source import (
    DEFAULT_BASE,
    FALLBACK_BASES,
    check_api,
    fetch_klines,
    get_ticker_price,
    get_futures_premium_index,
)
from mapeval.binance_ws import BinanceSpotWS


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
_MAX_HISTORY_FETCH_WORKERS: Final = 8


def _klines_to_df(klines: List[List]) -> pd.DataFrame:
    """Convert Binance kline arrays to a DataFrame with timestamp and close price."""
    if not klines:
        return pd.DataFrame(columns=["Date", "Close"])
    timestamps = [int(kline[0]) for kline in klines]
    closes = [float(kline[4]) for kline in klines]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="ms"),
            "Close": closes,
        }
    )


def _interval_to_freq(interval: str) -> str:
    return _INTERVAL_TO_FREQ.get(interval, "T")


def _fetch_symbol_history(
    symbol: str,
    interval: str,
    lookback: int,
    base_url: Sequence[str] | str,
) -> pd.DataFrame:
    try:
        klines = fetch_klines(symbol=symbol, interval=interval, limit=lookback, base_url=base_url)
    except Exception as exc:
        raise RuntimeError(f"Failed to bootstrap history for {symbol}: {exc}") from exc
    df = _klines_to_df(klines)
    if df.empty:
        raise ValueError(f"No kline data returned for {symbol}.")
    return df


class BaseMarketData:
    """Shared interface and methods for market data providers."""

    symbols: List[str]
    lookback: int
    price_history: pd.DataFrame

    def get_recent_window(self, rows: int = 120) -> pd.DataFrame:
        """Return the most recent slice of price history as a DataFrame with a Date column."""
        window = self.price_history.tail(rows).copy()
        window = window.reset_index()
        return window

    def latest_prices(self) -> Dict[str, float]:
        if self.price_history.empty:
            return {}
        last_row = self.price_history.iloc[-1]
        return {
            symbol: float(last_row[f"{symbol}_Close"])
            for symbol in self.symbols
            if f"{symbol}_Close" in last_row
        }


def _prepare_history_frame(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = history.rename(columns={"Close": f"{symbol}_Close"}).set_index("Date")
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    return frame


def _merge_history_frames(
    frames: Sequence[pd.DataFrame],
    columns: Sequence[str],
    lookback: int | None = None,
) -> pd.DataFrame:
    if not frames:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "Date"
        return empty
    merged = pd.concat(frames, axis=1, copy=False).sort_index()
    if getattr(merged.index, "tz", None) is not None:
        merged.index = merged.index.tz_convert(None)
    merged = merged[~merged.index.duplicated(keep="last")].ffill()
    if lookback is not None:
        merged = merged.tail(lookback)
    merged.index.name = "Date"
    return merged.reindex(columns=columns)


def _parallel_fetch_symbol_frames(
    symbols: Sequence[str],
    interval: str,
    lookback: int,
    base_urls: Sequence[str] | str,
) -> List[pd.DataFrame]:
    def fetch_for_symbol(symbol: str) -> pd.DataFrame:
        history = _fetch_symbol_history(symbol, interval, lookback, base_urls)
        return _prepare_history_frame(history, symbol)

    worker_count = min(len(symbols), _MAX_HISTORY_FETCH_WORKERS)
    if worker_count <= 1:
        return [fetch_for_symbol(symbol) for symbol in symbols]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fetch_for_symbol, symbols))


class RealTimeMarketData(BaseMarketData):
    """Maintain an up-to-date intraday price history for multiple symbols."""

    def __init__(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
        lookback: int = 500,
        base_url: Sequence[str] | str = DEFAULT_BASE,
    ) -> None:
        self.symbols = list(symbols)
        if not self.symbols:
            raise ValueError("At least one symbol must be provided.")
        self.interval = interval
        self.lookback = lookback
        self.base_urls = self._prepare_base_urls(base_url)
        self.base_url = self.base_urls[0]
        self._columns = [f"{symbol}_Close" for symbol in self.symbols]
        self.price_history = self._bootstrap_history()
        self._latest_prices_cache = self._extract_latest_prices(self.price_history)
        self._ws: Optional[BinanceSpotWS] = None
        self._funding_cache: Dict[str, float] = {}
        self._last_funding_fetch_ts: Optional[pd.Timestamp] = None
        self._consecutive_fetch_failures = 0
        self._offline_until: Optional[float] = None
        self._offline_backoff_seconds = 60.0

    @staticmethod
    def _prepare_base_urls(base_url: Sequence[str] | str) -> List[str]:
        if isinstance(base_url, (list, tuple, set)):
            candidates = list(base_url)
        else:
            focus = str(base_url).strip() or DEFAULT_BASE
            if focus.rstrip("/") == DEFAULT_BASE.rstrip("/"):
                candidates = list(FALLBACK_BASES)
            else:
                candidates = [focus]
        normalized: List[str] = []
        seen = set()
        for value in candidates:
            url = str(value).strip().rstrip("/")
            if not url or url in seen:
                continue
            normalized.append(url)
            seen.add(url)
        return normalized or [DEFAULT_BASE.rstrip("/")]

    def _bootstrap_history(self) -> pd.DataFrame:
        frames = _parallel_fetch_symbol_frames(
            self.symbols,
            self.interval,
            self.lookback,
            self.base_urls,
        )
        return _merge_history_frames(frames, self._columns, lookback=self.lookback)

    def _extract_latest_prices(self, history: pd.DataFrame) -> Dict[str, float]:
        if history.empty:
            return {}
        last_row = history.iloc[-1]
        latest: Dict[str, float] = {}
        for symbol in self.symbols:
            value = last_row.get(f"{symbol}_Close")
            if pd.notna(value):
                latest[symbol] = float(value)
        return latest

    @staticmethod
    def _normalize_timestamp(timestamp: pd.Timestamp | None = None) -> pd.Timestamp:
        normalized = pd.to_datetime(timestamp or pd.Timestamp.utcnow().floor("s"))
        if isinstance(normalized, pd.Timestamp) and normalized.tzinfo is not None:
            normalized = normalized.tz_convert(None)
        return normalized

    def refresh_history(self) -> None:
        """Force a history refresh from the remote API."""
        self.price_history = self._bootstrap_history()
        self._latest_prices_cache = self._extract_latest_prices(self.price_history)

    def fetch_latest_prices(self) -> Dict[str, float]:
        """Retrieve the most recent ticker price for all configured symbols."""
        # Prefer websocket snapshot if running and populated
        if self._ws is not None:
            ws_prices = self._ws.get_latest_prices()
            if all(sym in ws_prices for sym in self.symbols):
                return {sym: float(ws_prices[sym]) for sym in self.symbols}
        now = time.time()
        if self._offline_until is not None and now < self._offline_until:
            cached = self.latest_prices()
            if cached:
                return cached
        try:
            payload = get_ticker_price(symbols=self.symbols, base_url=self.base_urls)
            self._consecutive_fetch_failures = 0
            self._offline_until = None
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
            self._consecutive_fetch_failures += 1
            if self._consecutive_fetch_failures >= 3:
                cached = self.latest_prices()
                if cached:
                    self._offline_until = now + self._offline_backoff_seconds
                    return cached
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
        timestamp = self._normalize_timestamp(timestamp)
        history = self.price_history
        if history.empty:
            row = {
                f"{symbol}_Close": float(prices[symbol])
                for symbol in self.symbols
                if symbol in prices and prices[symbol] is not None
            }
            self.price_history = pd.DataFrame([row], index=pd.DatetimeIndex([timestamp], name="Date"))
            self.price_history = self.price_history.reindex(columns=self._columns)
            self._latest_prices_cache = self._extract_latest_prices(self.price_history)
            return

        last_timestamp = history.index[-1]
        if timestamp < last_timestamp:
            row = {
                f"{symbol}_Close": (
                    float(prices[symbol]) if symbol in prices and prices[symbol] is not None else float("nan")
                )
                for symbol in self.symbols
            }
            fallback = pd.DataFrame([row], index=pd.DatetimeIndex([timestamp], name="Date"))
            history = pd.concat([history, fallback], copy=False)
            history = history[~history.index.duplicated(keep="last")].sort_index().ffill()
            if len(history) > self.lookback:
                history = history.iloc[-self.lookback :].copy()
            self.price_history = history.reindex(columns=self._columns)
            self._latest_prices_cache = self._extract_latest_prices(self.price_history)
            return

        latest_prices = dict(self._latest_prices_cache)
        row = {
            f"{symbol}_Close": float(prices[symbol])
            if symbol in prices and prices[symbol] is not None
            else latest_prices.get(symbol, float("nan"))
            for symbol in self.symbols
        }
        history.loc[timestamp, self._columns] = [row[column] for column in self._columns]
        history.index.name = "Date"
        if len(history) > self.lookback:
            history = history.iloc[-self.lookback :].copy()
            self.price_history = history
        self._latest_prices_cache = {
            symbol: float(row[f"{symbol}_Close"])
            for symbol in self.symbols
            if pd.notna(row[f"{symbol}_Close"])
        }

    def get_recent_window(self, rows: int = 120) -> pd.DataFrame:
        """Return the most recent slice of price history as a DataFrame with a Date column."""
        window = self.price_history.tail(rows).copy()
        window = window.reset_index()
        return window

    def latest_prices(self) -> Dict[str, float]:
        return dict(self._latest_prices_cache)
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


class BacktestMarketData(BaseMarketData):
    """Simulate market data replay from a historical DataFrame."""

    def __init__(
        self,
        history_df: pd.DataFrame,
        symbols: Iterable[str],
        interval: str = "1m",
        lookback: int = 500,
    ) -> None:
        self.symbols = list(symbols)
        self.interval = interval
        self.lookback = lookback

        # Ensure history_df is sorted and has Date index
        self.full_history = history_df.sort_index()
        if getattr(self.full_history.index, "tz", None) is not None:
            self.full_history.index = self.full_history.index.tz_convert(None)

        self.current_idx = 0
        self.price_history = pd.DataFrame()

        # Initialize with enough data for lookback if possible
        if len(self.full_history) > lookback:
            self.current_idx = lookback
            self.price_history = self.full_history.iloc[: self.current_idx].copy()
        else:
            self.current_idx = 0
            self.price_history = pd.DataFrame()

    def fetch_latest_prices(self) -> Dict[str, float]:
        if self.current_idx >= len(self.full_history):
            raise StopIteration("Backtest finished.")

        row = self.full_history.iloc[self.current_idx]
        prices = {}
        for symbol in self.symbols:
            col = f"{symbol}_Close"
            if col in row:
                prices[symbol] = float(row[col])
        return prices

    def append_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp | None = None) -> None:
        # In backtest, advance the window as a O(1) slice of the pre-loaded history
        if self.current_idx < len(self.full_history):
            end_idx = self.current_idx + 1
            start_idx = max(0, end_idx - self.lookback)
            self.price_history = self.full_history.iloc[start_idx:end_idx]
            self.current_idx += 1

    def refresh_funding_rates(self, throttle_seconds: int = 60) -> Dict[str, float]:
        return {}

    def start_websocket(self) -> None:
        pass

    def stop_websocket(self) -> None:
        pass


def load_historical_data(
    symbols: Iterable[str],
    interval: str,
    lookback: int,
    base_url: Sequence[str] | str = DEFAULT_BASE,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch and merge historical data for multiple symbols."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading historical data from {cache_path}...")
        try:
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            # Ensure columns match requested symbols
            expected_cols = [f"{s}_Close" for s in symbols]
            missing = [c for c in expected_cols if c not in df.columns]
            if not missing:
                return df
            print(f"Cached data missing columns {missing}, refetching...")
        except Exception as e:
            print(f"Failed to load cache: {e}, refetching...")

    base_urls = RealTimeMarketData._prepare_base_urls(base_url)
    symbols = list(symbols)
    columns = [f"{symbol}_Close" for symbol in symbols]

    frames = _parallel_fetch_symbol_frames(symbols, interval, lookback, base_urls)
    
    if not frames:
        return pd.DataFrame()
        
    merged = _merge_history_frames(frames, columns)

    if cache_path and not merged.empty:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            merged.to_csv(cache_path)
            print(f"Saved historical data to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    return merged
