"""Tests for refactored shared utilities: exposure_utils, order_executor base, data_manager base."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from mapeval.data_manager import BacktestMarketData
from mapeval.exposure_utils import compute_fallback_exposures, sanitize_exposures
from mapeval.order_executor import (
    PaperExecutor,
    SimulatedExecutorBase,
    SimulationExecutor,
    create_executor,
)
from mapeval.order_models import Order, OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# exposure_utils.sanitize_exposures
# ---------------------------------------------------------------------------

class TestSanitizeExposures:
    def test_clips_per_symbol(self):
        notes: list[str] = []
        result = sanitize_exposures(
            exposures={"BTCUSDT": 10.0, "ETHUSDT": -8.0},
            symbols=["BTCUSDT", "ETHUSDT"],
            per_symbol_max_exposure=5.0,
            max_exposure_delta=50.0,
            max_leverage=30.0,
            last_exposures={"BTCUSDT": 0.0, "ETHUSDT": 0.0},
            sanitization_notes=notes,
        )
        assert result is not None
        assert result["BTCUSDT"] == pytest.approx(5.0)
        assert result["ETHUSDT"] == pytest.approx(-5.0)
        assert any("clipped" in n for n in notes)

    def test_delta_limiting(self):
        notes: list[str] = []
        result = sanitize_exposures(
            exposures={"BTCUSDT": 5.0},
            symbols=["BTCUSDT"],
            per_symbol_max_exposure=10.0,
            max_exposure_delta=2.0,
            max_leverage=30.0,
            last_exposures={"BTCUSDT": 1.0},
            sanitization_notes=notes,
        )
        assert result is not None
        assert result["BTCUSDT"] == pytest.approx(3.0)

    def test_scales_total_leverage(self):
        notes: list[str] = []
        result = sanitize_exposures(
            exposures={"BTCUSDT": 8.0, "ETHUSDT": 8.0},
            symbols=["BTCUSDT", "ETHUSDT"],
            per_symbol_max_exposure=10.0,
            max_exposure_delta=10.0,
            max_leverage=10.0,
            last_exposures={"BTCUSDT": 0.0, "ETHUSDT": 0.0},
            sanitization_notes=notes,
        )
        assert result is not None
        total = abs(result["BTCUSDT"]) + abs(result["ETHUSDT"])
        assert total == pytest.approx(10.0)

    def test_returns_none_for_non_numeric(self):
        notes: list[str] = []
        result = sanitize_exposures(
            exposures={"BTCUSDT": "invalid"},
            symbols=["BTCUSDT"],
            per_symbol_max_exposure=10.0,
            max_exposure_delta=10.0,
            max_leverage=10.0,
            last_exposures={"BTCUSDT": 0.0},
            sanitization_notes=notes,
        )
        assert result is None

    def test_zero_leverage_zeros_all(self):
        notes: list[str] = []
        result = sanitize_exposures(
            exposures={"BTCUSDT": 5.0},
            symbols=["BTCUSDT"],
            per_symbol_max_exposure=10.0,
            max_exposure_delta=10.0,
            max_leverage=0.0,
            last_exposures={"BTCUSDT": 0.0},
            sanitization_notes=notes,
        )
        assert result is not None
        assert result["BTCUSDT"] == 0.0


# ---------------------------------------------------------------------------
# exposure_utils.compute_fallback_exposures
# ---------------------------------------------------------------------------

class MockTools:
    """Mock tools for testing fallback exposures."""
    def calculate_moving_average(self, symbol: str, time: pd.Timestamp, window: int) -> float:
        if window == 21:
            return 55000.0
        if window == 63:
            return 50000.0
        return None

    def calculate_volatility(self, symbol: str, time: pd.Timestamp, window: int) -> float:
        return 0.02


class TestComputeFallbackExposures:
    def test_returns_dict_for_all_symbols(self):
        result = compute_fallback_exposures(
            symbols=["BTCUSDT", "ETHUSDT"],
            current_time=pd.Timestamp("2024-01-01"),
            available_tools=MockTools(),
            max_leverage=10.0,
        )
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    def test_momentum_direction(self):
        result = compute_fallback_exposures(
            symbols=["BTCUSDT"],
            current_time=pd.Timestamp("2024-01-01"),
            available_tools=MockTools(),
            max_leverage=10.0,
        )
        # short_ma > long_ma => positive momentum => positive exposure
        assert result["BTCUSDT"] > 0


# ---------------------------------------------------------------------------
# SimulatedExecutorBase shared fill logic
# ---------------------------------------------------------------------------

class TestSimulatedExecutorBase:
    def test_simulation_inherits_base(self):
        executor = SimulationExecutor()
        assert isinstance(executor, SimulatedExecutorBase)

    def test_paper_inherits_base(self):
        executor = PaperExecutor()
        assert isinstance(executor, SimulatedExecutorBase)

    def test_sim_submit_order_fills(self):
        executor = SimulationExecutor(commission_rate=0.001, slippage=0.001)
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
        )
        result = executor.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.exchange_order_id.startswith("SIM-")
        assert result.avg_fill_price > 50000.0  # slippage applied

    def test_paper_submit_order_fills(self):
        executor = PaperExecutor(commission_rate=0.001, slippage=0.001, fill_latency_ms=0)
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
        )
        result = executor.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.exchange_order_id.startswith("PAPER-")
        assert result.avg_fill_price < 50000.0  # sell slippage

    def test_reject_on_invalid_price(self):
        executor = SimulationExecutor()
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=0.0,
        )
        result = executor.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_sync_positions_returns_copy(self):
        executor = SimulationExecutor()
        positions = executor.sync_positions()
        assert isinstance(positions, dict)
        assert positions == {}

    def test_sync_balance(self):
        executor = SimulationExecutor(initial_balance=50_000.0)
        assert executor.sync_balance() == 50_000.0

    def test_create_executor_factory(self):
        sim = create_executor("simulation")
        assert sim.get_execution_mode() == "simulation"
        paper = create_executor("paper")
        assert paper.get_execution_mode() == "paper"


# ---------------------------------------------------------------------------
# BaseMarketData (via BacktestMarketData)
# ---------------------------------------------------------------------------

class TestBaseMarketData:
    def _make_backtest_data(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        df = pd.DataFrame(
            {"BTCUSDT_Close": range(50000, 50020), "ETHUSDT_Close": range(3000, 3020)},
            index=dates,
        )
        df.index.name = "Date"
        return BacktestMarketData(df, symbols=["BTCUSDT", "ETHUSDT"], lookback=10)

    def test_latest_prices(self):
        data = self._make_backtest_data()
        prices = data.latest_prices()
        assert "BTCUSDT" in prices
        assert "ETHUSDT" in prices

    def test_latest_prices_empty(self):
        df = pd.DataFrame()
        data = BacktestMarketData(df, symbols=["BTCUSDT"], lookback=10)
        assert data.latest_prices() == {}

    def test_get_recent_window(self):
        data = self._make_backtest_data()
        window = data.get_recent_window(5)
        assert "Date" in window.columns
        assert len(window) <= 5
