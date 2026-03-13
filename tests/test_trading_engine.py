"""Unit tests for trading_engine.py core logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from mapeval.order_executor import GuardedOrderExecutor, PaperExecutor
from mapeval.order_models import Order, OrderSide, OrderType
from mapeval.security import ReadOnlyGuard
from mapeval.trading_engine import AccountState, FuturesPosition, RealTimeTradingEngine


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


class RecordingExecutor:
    def __init__(self) -> None:
        self.orders = []

    def submit_order(self, order):
        self.orders.append(order)
        from mapeval.order_models import OrderResult, OrderStatus
        return OrderResult(
            order=order,
            status=OrderStatus.FILLED,
            exchange_order_id="paper-1",
            filled_quantity=order.quantity,
            avg_fill_price=order.price or 0.0,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    def get_order_status(self, symbol: str, order_id: str):
        from mapeval.order_models import OrderStatus
        return OrderStatus.FILLED

    def sync_positions(self):
        return {}

    def sync_balance(self) -> float:
        return 0.0

    def get_execution_mode(self) -> str:
        return "paper"


class ReconciliationExecutor(RecordingExecutor):
    def __init__(self, remote_balance: float) -> None:
        super().__init__()
        self.remote_balance = remote_balance

    def sync_balance(self) -> float:
        return self.remote_balance


class EmptyRemoteExecutor(RecordingExecutor):
    def sync_balance(self) -> float:
        return 0.0


class StubAuditLogger:
    def __init__(self) -> None:
        self.entries = []

    def log_control_action(self, action: str, details=None, execution_mode: str = "live") -> None:
        self.entries.append({
            "action": action,
            "details": details or {},
            "execution_mode": execution_mode,
        })


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


class TestExecutionSafety:
    def test_paper_executor_uses_initial_balance(self):
        executor = PaperExecutor(initial_balance=12_345.0)
        assert executor.sync_balance() == pytest.approx(12_345.0)

    def test_guarded_executor_blocks_mutations_in_read_only_mode(self):
        guard = ReadOnlyGuard(enabled=True)
        executor = GuardedOrderExecutor(PaperExecutor(initial_balance=10_000.0), guard)
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=50_000.0,
        )

        with pytest.raises(PermissionError):
            executor.submit_order(order)

    def test_engine_uses_executor_for_paper_mode(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        agent = MockAgent()
        executor = RecordingExecutor()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode="paper",
            order_executor=executor,
        )

        engine._rebalance_position("BTCUSDT", 0.1, 50_000.0, pd.Timestamp.utcnow())

        assert len(executor.orders) == 1
        assert executor.orders[0].symbol == "BTCUSDT"

    def test_kill_switch_enables_read_only(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        engine.read_only_guard = ReadOnlyGuard(enabled=False)

        result = engine.activate_kill_switch(reason="test")

        assert result["kill_switch_active"] is True
        assert engine.read_only_guard.is_read_only is True

    def test_reconcile_reports_balance_discrepancy(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode="paper",
            order_executor=ReconciliationExecutor(remote_balance=90_000.0),
        )

        report = engine.reconcile()

        assert report["status"] == "completed"
        assert len(report["discrepancies"]) == 1
        assert report["discrepancies"][0]["type"] == "balance"

    def test_control_actions_are_audited(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode="paper",
        )
        engine.read_only_guard = ReadOnlyGuard(enabled=False)
        engine.audit_logger = StubAuditLogger()

        engine.set_read_only(True)
        engine.activate_kill_switch(reason="test")
        engine.release_kill_switch()

        actions = [entry["action"] for entry in engine.audit_logger.entries]
        assert "read_only_enabled" in actions
        assert "kill_switch_activated" in actions
        assert "kill_switch_released" in actions

    def test_sync_preserves_local_state_on_empty_remote_snapshot(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode="paper",
            order_executor=EmptyRemoteExecutor(),
        )
        engine.account.positions["BTCUSDT"] = FuturesPosition(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50_000.0,
            leverage=1.0,
            opened_at=pd.Timestamp.utcnow(),
        )

        engine._sync_account_from_executor(pd.Timestamp.utcnow(), {"BTCUSDT": 50_000.0})

        assert "BTCUSDT" in engine.account.positions
