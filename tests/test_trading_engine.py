"""Unit tests for trading_engine.py core logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from trading_engine import AccountState, FuturesPosition, RealTimeTradingEngine


class MockMarketData:
    def __init__(self, symbols: list[str], prices: dict[str, float]) -> None:
        self.symbols = symbols
        self._prices = prices

    def fetch_latest_prices(self) -> dict[str, float]:
        return dict(self._prices)

    def latest_prices(self) -> dict[str, float]:
        return dict(self._prices)

    def append_prices(self, prices: dict[str, float], timestamp=None) -> None:
        self._prices.update(prices)

    def get_recent_window(self, rows: int = 120) -> pd.DataFrame:
        return pd.DataFrame()

    def refresh_funding_rates(self, throttle_seconds: int = 60) -> dict[str, float]:
        return {}


class MockAgent:
    def __init__(self) -> None:
        self.last_reasoning = ""
        self.last_sanitization_notes = []

    def generate_trading_signal(self, current_time, market_data_slice, tools):
        return {}


class TestAccountState:
    def test_initial_equity_equals_balance(self):
        account = AccountState(balance=100_000.0)
        assert account.equity == 100_000.0
        assert account.available_margin == 100_000.0

    def test_mark_to_market_with_no_positions(self):
        account = AccountState(balance=50_000.0)
        account.mark_to_market({"BTCUSDT": 50000.0}, max_leverage=10.0)
        assert account.unrealized_pnl == 0.0
        assert account.equity == 50_000.0

    def test_mark_to_market_with_long_position_profit(self):
        account = AccountState(balance=10_000.0)
        account.positions["BTCUSDT"] = FuturesPosition(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            leverage=5.0,
            opened_at=pd.Timestamp.utcnow(),
        )
        account.mark_to_market({"BTCUSDT": 55000.0}, max_leverage=10.0)
        expected_pnl = (55000.0 - 50000.0) * 0.1
        assert account.unrealized_pnl == pytest.approx(expected_pnl)
        assert account.equity == pytest.approx(10_000.0 + expected_pnl)

    def test_mark_to_market_with_long_position_loss(self):
        account = AccountState(balance=10_000.0)
        account.positions["ETHUSDT"] = FuturesPosition(
            symbol="ETHUSDT",
            quantity=1.0,
            entry_price=3000.0,
            leverage=3.0,
            opened_at=pd.Timestamp.utcnow(),
        )
        account.mark_to_market({"ETHUSDT": 2800.0}, max_leverage=10.0)
        expected_pnl = (2800.0 - 3000.0) * 1.0
        assert account.unrealized_pnl == pytest.approx(expected_pnl)
        assert account.equity == pytest.approx(10_000.0 + expected_pnl)

    def test_mark_to_market_with_short_position(self):
        account = AccountState(balance=10_000.0)
        account.positions["BTCUSDT"] = FuturesPosition(
            symbol="BTCUSDT",
            quantity=-0.1,
            entry_price=50000.0,
            leverage=5.0,
            opened_at=pd.Timestamp.utcnow(),
        )
        account.mark_to_market({"BTCUSDT": 48000.0}, max_leverage=10.0)
        expected_pnl = (48000.0 - 50000.0) * (-0.1)
        assert account.unrealized_pnl == pytest.approx(expected_pnl)


class TestRealTimeTradingEngine:
    def test_validate_exposures_rejects_unknown_symbol(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        result = engine._validate_exposures(
            {"BTCUSDT": 1.0, "UNKNOWN": 0.5},
            allow_rescale=True,
        )
        assert result["valid"] is False
        assert result["code"] == "UNKNOWN_SYMBOL"

    def test_validate_exposures_clips_per_symbol_limit(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            per_symbol_max_exposure=5.0,
        )
        result = engine._validate_exposures(
            {"BTCUSDT": 8.0},
            allow_rescale=True,
        )
        assert result["valid"] is True
        assert result["exposures"]["BTCUSDT"] == pytest.approx(5.0)
        assert len(result["notes"]) > 0

    def test_validate_exposures_scales_total_leverage(self):
        market_data = MockMarketData(["BTCUSDT", "ETHUSDT"], {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        result = engine._validate_exposures(
            {"BTCUSDT": 8.0, "ETHUSDT": 8.0},
            allow_rescale=True,
        )
        assert result["valid"] is True
        total = abs(result["exposures"]["BTCUSDT"]) + abs(result["exposures"]["ETHUSDT"])
        assert total == pytest.approx(10.0)

    def test_current_exposures_empty_positions(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        exposures = engine._current_exposures({"BTCUSDT": 50000.0})
        assert exposures == {"BTCUSDT": 0.0}

    def test_margin_requirement_calculation(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        margin = engine._margin_requirement({"BTCUSDT": 5.0}, equity=100_000.0)
        assert margin == pytest.approx(50_000.0)


class TestLiquidationDetection:
    def test_check_liquidation_when_equity_positive(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        engine.account.equity = 50_000.0
        result = engine._check_liquidation({"BTCUSDT": 50000.0}, pd.Timestamp.utcnow())
        assert result is False

    def test_check_liquidation_when_equity_zero(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        engine.account.equity = 0.0
        result = engine._check_liquidation({"BTCUSDT": 50000.0}, pd.Timestamp.utcnow())
        assert result is True

