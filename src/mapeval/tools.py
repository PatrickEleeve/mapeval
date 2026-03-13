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
        """Compute the Relative Strength Index (RSI)."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size + 1)
        if len(window) <= 1:
            return None
        deltas = window.diff().dropna()
        if deltas.empty:
            return None
        gains = deltas.clip(lower=0.0)
        losses = -deltas.clip(upper=0.0)
        avg_gain = gains.rolling(window_size).mean().iloc[-1]
        avg_loss = losses.rolling(window_size).mean().iloc[-1]
        if pd.isna(avg_gain) or pd.isna(avg_loss):
            return None
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(max(0.0, min(100.0, rsi)))

    def calculate_macd(
        self,
        asset: str,
        end_date: pd.Timestamp,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Optional[Dict[str, float]]:
        """Compute MACD line, signal line, and histogram."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        closes = series.loc[:end]
        if closes.empty:
            return None
        if len(closes) < slow_period + signal_period:
            return None
        ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
        }

    def calculate_atr(
        self,
        asset: str,
        end_date: pd.Timestamp,
        window_size: int = 14,
    ) -> Optional[float]:
        """Approximate Average True Range (ATR) using close-to-close changes."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size + 1)
        if len(window) <= 1:
            return None
        true_ranges = window.diff().abs().dropna()
        if true_ranges.empty:
            return 0.0
        atr_series = true_ranges.rolling(window_size).mean().dropna()
        if atr_series.empty:
            return float(true_ranges.mean())
        return float(atr_series.iloc[-1])

    def calculate_bollinger_bands(
        self,
        asset: str,
        end_date: pd.Timestamp,
        window_size: int = 20,
        num_std: float = 2.0,
    ) -> Optional[Dict[str, float]]:
        """Return Bollinger Bands mid/upper/lower values."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size)
        if len(window) < window_size:
            return None
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        upper = mean + num_std * std
        lower = mean - num_std * std
        bandwidth = ((upper - lower) / mean) if mean != 0 else None
        return {
            "mid": mean,
            "upper": upper,
            "lower": lower,
            "bandwidth": float(bandwidth) if bandwidth is not None else None,
        }

    def calculate_coefficient_of_variation(
        self,
        asset: str,
        end_date: pd.Timestamp,
        window_size: int = 20,
    ) -> Optional[float]:
        """Coefficient of Variation (std/mean) over the window."""
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        window = series.loc[:end].tail(window_size)
        if window.empty:
            return None
        mean = float(window.mean())
        if mean == 0.0:
            return None
        std = float(window.std(ddof=0))
        return std / mean

    def calculate_moving_average_slope(
        self,
        asset: str,
        end_date: pd.Timestamp,
        window_size: int,
        periods: int = 5,
    ) -> Optional[float]:
        """Estimate slope of the moving average in units of price change per period."""
        if periods <= 0:
            raise ValueError("periods must be positive")
        end = pd.to_datetime(end_date)
        series = self._resolve_series(asset)
        ma_series = series.loc[:end].rolling(window=window_size).mean().dropna()
        if len(ma_series) <= periods:
            return None
        current = float(ma_series.iloc[-1])
        previous = float(ma_series.iloc[-(periods + 1)])
        return (current - previous) / periods
