"""Financial utility functions shared across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class FinancialTools:
    """Expose simple analytics over the market data set."""

    market_data_df: pd.DataFrame
    funding_rates: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        data = self.market_data_df.copy()
        if "Date" in data.columns:
            date_series = pd.to_datetime(data["Date"])
            if getattr(date_series.dt, "tz", None) is not None:
                date_series = date_series.dt.tz_convert(None)
            data["Date"] = date_series
            data = data.set_index("Date")
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_convert(None)
        self.market_data = data.sort_index()

    def get_funding_rate(self, asset: str) -> Optional[float]:
        if not self.funding_rates:
            return None
        return self.funding_rates.get(asset)

    def _resolve_series(self, asset: str) -> pd.Series:
        column = f"{asset}_Close"
        if column not in self.market_data.columns:
            raise ValueError(f"Unknown asset column: {column}")
        return self.market_data[column]

    def get_historical_prices(self, asset: str, end_date: pd.Timestamp, days: int) -> pd.Series:
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(days)
        return window

    def calculate_moving_average(self, asset: str, end_date: pd.Timestamp, window_size: int) -> Optional[float]:
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size)
        if window.empty:
            return None
        return float(window.mean())

    def calculate_volatility(self, asset: str, end_date: pd.Timestamp, window_size: int) -> Optional[float]:
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size)
        if window.empty:
            return None
        returns = window.pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.std())

    def calculate_rsi(self, asset: str, end_date: pd.Timestamp, window_size: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        # Need enough data to stabilize EMA
        lookback = window_size * 4
        subset = series.loc[:end].tail(lookback)
        if len(subset) < window_size + 1:
            return None

        delta = subset.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / window_size, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window_size, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def calculate_macd(
        self, asset: str, end_date: pd.Timestamp, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Optional[Dict[str, float]]:
        """Calculate MACD, Signal line, and Histogram."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        lookback = slow * 4
        subset = series.loc[:end].tail(lookback)
        if len(subset) < slow:
            return None

        ema_fast = subset.ewm(span=fast, adjust=False).mean()
        ema_slow = subset.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line

        return {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "hist": float(hist.iloc[-1]),
        }
