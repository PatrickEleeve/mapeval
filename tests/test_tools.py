"""Unit tests for tools.py financial indicators."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from tools import FinancialTools


def make_price_df(prices: list[float], symbol: str = "BTCUSDT") -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=max(len(prices), 1), freq="1min")
    dates = dates.tz_localize(None)
    return pd.DataFrame({
        "Date": dates,
        f"{symbol}_Close": prices if prices else [float("nan")],
    })


class TestMovingAverage:
    def test_basic_moving_average(self):
        df = make_price_df([100.0, 110.0, 120.0, 130.0, 140.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        ma = tools.calculate_moving_average("BTCUSDT", end_date, window_size=3)
        assert ma == pytest.approx((120.0 + 130.0 + 140.0) / 3)

    def test_moving_average_full_window(self):
        df = make_price_df([50.0, 60.0, 70.0, 80.0, 90.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        ma = tools.calculate_moving_average("BTCUSDT", end_date, window_size=5)
        assert ma == pytest.approx(70.0)

    def test_moving_average_empty_returns_none(self):
        df = pd.DataFrame({
            "Date": pd.DatetimeIndex([], dtype="datetime64[ns]"),
            "BTCUSDT_Close": pd.Series([], dtype=float),
        })
        tools = FinancialTools(df)
        end_date = pd.Timestamp.now()
        ma = tools.calculate_moving_average("BTCUSDT", end_date, window_size=5)
        assert ma is None


class TestVolatility:
    def test_volatility_calculation(self):
        df = make_price_df([100.0, 102.0, 98.0, 105.0, 103.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        vol = tools.calculate_volatility("BTCUSDT", end_date, window_size=5)
        assert vol is not None
        assert vol > 0

    def test_volatility_constant_prices(self):
        df = make_price_df([100.0, 100.0, 100.0, 100.0, 100.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        vol = tools.calculate_volatility("BTCUSDT", end_date, window_size=5)
        assert vol == pytest.approx(0.0)


class TestRSI:
    def test_rsi_overbought(self):
        prices = [100.0 + i * 2 for i in range(20)]
        df = make_price_df(prices)
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        rsi = tools.calculate_rsi("BTCUSDT", end_date, window_size=14)
        assert rsi is not None
        assert rsi > 70

    def test_rsi_oversold(self):
        prices = [200.0 - i * 2 for i in range(20)]
        df = make_price_df(prices)
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        rsi = tools.calculate_rsi("BTCUSDT", end_date, window_size=14)
        assert rsi is not None
        assert rsi < 30

    def test_rsi_bounds(self):
        df = make_price_df([100.0, 105.0, 102.0, 108.0, 106.0, 110.0, 107.0, 112.0,
                            109.0, 115.0, 112.0, 118.0, 115.0, 120.0, 117.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        rsi = tools.calculate_rsi("BTCUSDT", end_date, window_size=14)
        assert rsi is not None
        assert 0 <= rsi <= 100


class TestMACD:
    def test_macd_returns_dict(self):
        prices = [100.0 + i * 0.5 for i in range(50)]
        df = make_price_df(prices)
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        macd = tools.calculate_macd("BTCUSDT", end_date)
        assert macd is not None
        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd

    def test_macd_insufficient_data(self):
        df = make_price_df([100.0, 101.0, 102.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        macd = tools.calculate_macd("BTCUSDT", end_date)
        assert macd is None


class TestBollingerBands:
    def test_bollinger_bands_structure(self):
        prices = [100.0 + i % 5 for i in range(30)]
        df = make_price_df(prices)
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        bands = tools.calculate_bollinger_bands("BTCUSDT", end_date, window_size=20)
        assert bands is not None
        assert "upper" in bands
        assert "mid" in bands
        assert "lower" in bands
        assert bands["upper"] > bands["mid"] > bands["lower"]

    def test_bollinger_bands_insufficient_data(self):
        df = make_price_df([100.0, 101.0, 102.0])
        tools = FinancialTools(df)
        end_date = df["Date"].iloc[-1]
        bands = tools.calculate_bollinger_bands("BTCUSDT", end_date, window_size=20)
        assert bands is None


class TestFundingRate:
    def test_get_funding_rate_exists(self):
        df = make_price_df([100.0])
        tools = FinancialTools(df, funding_rates={"BTCUSDT": 0.0001})
        rate = tools.get_funding_rate("BTCUSDT")
        assert rate == pytest.approx(0.0001)

    def test_get_funding_rate_missing(self):
        df = make_price_df([100.0])
        tools = FinancialTools(df, funding_rates={"BTCUSDT": 0.0001})
        rate = tools.get_funding_rate("ETHUSDT")
        assert rate is None

    def test_get_funding_rate_no_data(self):
        df = make_price_df([100.0])
        tools = FinancialTools(df)
        rate = tools.get_funding_rate("BTCUSDT")
        assert rate is None
