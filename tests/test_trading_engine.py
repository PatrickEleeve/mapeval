"""Unit tests for trading_engine.py core logic."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from mapeval.data_manager import BacktestMarketData
from mapeval.order_executor import GuardedOrderExecutor, PaperExecutor
from mapeval.order_models import Order, OrderSide, OrderType, PositionInfo
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
        self.calls = 0

    def generate_trading_signal(self, current_time, market_data_slice, tools):
        self.calls += 1
        return {}


class LongAgent(MockAgent):
    def generate_trading_signal(self, current_time, market_data_slice, tools):
        self.calls += 1
        return {"BTCUSDT": 1.0}


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


class FailingExecutor(RecordingExecutor):
    def submit_order(self, order):
        raise RuntimeError("exchange unavailable")


class ResidualPositionExecutor(RecordingExecutor):
    def sync_positions(self):
        return {
            "BTCUSDT": PositionInfo(
                symbol="BTCUSDT",
                quantity=0.25,
                entry_price=100.0,
                mark_price=97.0,
                unrealized_pnl=-0.75,
                leverage=1.0,
            )
        }

    def sync_balance(self) -> float:
        return 1_000.0


class StubAuditLogger:
    def __init__(self) -> None:
        self.entries = []

    def log_control_action(self, action: str, details=None, execution_mode: str = "live") -> None:
        self.entries.append(
            {
                "action": action,
                "details": details or {},
                "execution_mode": execution_mode,
            }
        )

    def log_order(self, **entry) -> None:
        self.entries.append(entry)


class RecordingEventBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event) -> None:
        self.events.append(event)


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
    @pytest.mark.parametrize(("stop_enabled", "expected_stop_closes"), [(False, 0), (True, 1)])
    def test_backtest_stop_loss_is_explicit_and_deterministic(
        self, stop_enabled, expected_stop_closes
    ):
        index = pd.date_range("2024-01-01", periods=4, freq="min", name="Date")
        market_data = BacktestMarketData(
            pd.DataFrame({"BTCUSDT_Close": [100.0, 100.0, 97.0, 97.0]}, index=index),
            symbols=["BTCUSDT"],
            interval="1m",
            lookback=1,
        )
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=LongAgent(),
            initial_capital=1_000.0,
            max_leverage=1.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            stop_loss_enabled=stop_enabled,
        )

        summary = engine.run(duration_seconds=180.0, replay_interval_seconds=60.0)

        stop_closes = [
            trade for trade in summary["trade_log"] if trade.get("exit_reason") == "stop_loss"
        ]
        assert len(stop_closes) == expected_stop_closes

    def test_historical_replay_advances_once_per_bar_without_sleep(self, monkeypatch):
        index = pd.date_range("2024-01-01", periods=5, freq="min", name="Date")
        history = pd.DataFrame(
            {"BTCUSDT_Close": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=index,
        )
        market_data = BacktestMarketData(
            history,
            symbols=["BTCUSDT"],
            interval="1m",
            lookback=2,
        )
        agent = MockAgent()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=agent,
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )

        def fail_if_called(_seconds):
            pytest.fail("historical replay must not sleep")

        monkeypatch.setattr("mapeval.trading_engine.time.sleep", fail_if_called)

        summary = engine.run(duration_seconds=180.0, replay_interval_seconds=60.0)

        assert market_data.current_idx == len(history)
        assert agent.calls == 3
        assert len(summary["equity_history"]) == 3
        assert summary["equity_history"][0]["timestamp"] == index[2]

    def test_run_closes_positions_before_building_final_summary(self):
        index = pd.date_range("2024-01-01", periods=3, freq="min", name="Date")
        history = pd.DataFrame(
            {"BTCUSDT_Close": [100.0, 100.0, 100.0]},
            index=index,
        )
        market_data = BacktestMarketData(
            history,
            symbols=["BTCUSDT"],
            interval="1m",
            lookback=1,
        )
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=LongAgent(),
            initial_capital=1_000.0,
            max_leverage=1.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=120.0,
            commission_rate=0.001,
        )

        summary = engine.run(duration_seconds=120.0, replay_interval_seconds=60.0)

        final_account = summary["final_account"]
        trade_pnl = sum(entry["realized_pnl"] for entry in summary["trade_log"])
        assert engine.account.positions == {}
        assert final_account["unrealized_pnl"] == 0.0
        assert final_account["realized_pnl"] == pytest.approx(trade_pnl)
        assert final_account["equity"] == pytest.approx(1_000.0 + trade_pnl)
        assert summary["equity_history"][-1]["equity"] == pytest.approx(final_account["equity"])

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

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_validate_exposures_rejects_non_finite_values(self, value):
        engine = RealTimeTradingEngine(
            market_data=MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0}),
            agent=MockAgent(),
            initial_capital=100_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )

        result = engine._validate_exposures({"BTCUSDT": value}, allow_rescale=False)

        assert result["valid"] is False
        assert result["code"] == "INVALID_NUMBER"

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
        market_data = MockMarketData(
            ["BTCUSDT", "ETHUSDT"], {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
        )
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

    def test_realized_pnl_includes_open_and_close_commissions(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 100.0})
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=MockAgent(),
            initial_capital=1_000.0,
            max_leverage=10.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            commission_rate=0.001,
        )
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")

        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)
        engine._rebalance_position("BTCUSDT", 0.0, 100.0, timestamp)

        trade_pnl = sum(entry["realized_pnl"] for entry in engine.trade_log)
        assert engine.account.balance == pytest.approx(999.8)
        assert engine.account.realized_pnl == pytest.approx(-0.2)
        assert engine.account.realized_pnl == pytest.approx(trade_pnl)


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
    def _stop_engine(self, *, execution_mode="simulation", executor=None):
        return RealTimeTradingEngine(
            market_data=MockMarketData(["BTCUSDT"], {"BTCUSDT": 100.0}),
            agent=MockAgent(),
            initial_capital=1_000.0,
            max_leverage=2.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode=execution_mode,
            order_executor=executor,
            stop_loss_enabled=True,
        )

    def test_stop_lifecycle_preserves_protection_on_increase_and_reduce(self):
        engine = self._stop_engine()
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)
        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") == pytest.approx(98.0)

        engine.stop_loss_manager.update_trailing_stop("BTCUSDT", 105.0, "long", None)
        tightened = engine.stop_loss_manager.get_stop_price("BTCUSDT")
        engine._rebalance_position("BTCUSDT", 2.0, 100.0, timestamp)
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)

        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") == tightened

    def test_stop_lifecycle_reinitializes_on_reverse_and_removes_on_close(self):
        engine = self._stop_engine()
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)
        engine._rebalance_position("BTCUSDT", -1.0, 100.0, timestamp)
        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") == pytest.approx(102.0)

        engine._rebalance_position("BTCUSDT", 0.0, 100.0, timestamp)
        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") is None

    def test_triggered_stop_closes_once_with_reduce_only_and_exit_reason(self):
        executor = RecordingExecutor()
        engine = self._stop_engine(execution_mode="paper", executor=executor)
        engine.audit_logger = StubAuditLogger()
        engine.event_bus = RecordingEventBus()
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)

        first = engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp)
        second = engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp)

        assert first == ["BTCUSDT"]
        assert second == []
        assert len(executor.orders) == 2
        assert executor.orders[-1].reduce_only is True
        assert engine.trade_log[-1]["exit_reason"] == "stop_loss"
        assert engine.account.positions == {}
        assert engine.audit_logger.entries[-1]["action"] == "stop_loss_close"
        assert [event.event_type.value for event in engine.event_bus.events].count(
            "STOP_TRIGGERED"
        ) == 1

    def test_read_only_stop_alerts_then_closes_after_release(self):
        executor = RecordingExecutor()
        engine = self._stop_engine(execution_mode="paper", executor=executor)
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)
        engine.read_only_guard = ReadOnlyGuard(enabled=True)

        engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp)
        engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp + pd.Timedelta(seconds=30))
        assert len(executor.orders) == 1
        assert "BTCUSDT" in engine.account.positions
        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") is not None

        engine.read_only_guard.disable()
        engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp + pd.Timedelta(seconds=31))
        assert len(executor.orders) == 2
        assert engine.account.positions == {}

    def test_stop_execution_failure_activates_kill_switch(self):
        engine = self._stop_engine(execution_mode="paper", executor=FailingExecutor())
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine.account.positions["BTCUSDT"] = FuturesPosition("BTCUSDT", 1.0, 100.0, 1.0, timestamp)
        engine.stop_loss_manager.calculate_initial_stop("BTCUSDT", 100.0, "long", None, timestamp)
        engine.read_only_guard = ReadOnlyGuard(enabled=False)

        engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp)

        assert engine._kill_switch_active is True
        assert engine.read_only_guard.is_read_only is True
        assert "BTCUSDT" in engine.account.positions

    def test_residual_position_after_stop_keeps_protection_and_activates_kill_switch(self):
        executor = ResidualPositionExecutor()
        engine = self._stop_engine(execution_mode="paper", executor=executor)
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine._rebalance_position("BTCUSDT", 1.0, 100.0, timestamp)
        engine.read_only_guard = ReadOnlyGuard(enabled=False)

        engine._process_stop_losses({"BTCUSDT": 97.0}, timestamp)

        assert engine._kill_switch_active is True
        assert engine.account.positions["BTCUSDT"].quantity == pytest.approx(0.25)
        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") is not None

    def test_reconciled_position_gets_stop_on_next_price_update(self):
        engine = self._stop_engine()
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        engine.account.positions["BTCUSDT"] = FuturesPosition("BTCUSDT", 1.0, 100.0, 1.0, timestamp)

        engine._process_stop_losses({"BTCUSDT": 100.0}, timestamp)

        assert engine.stop_loss_manager.get_stop_price("BTCUSDT") == pytest.approx(98.0)

    def test_plan_preview_never_places_orders_or_records_decision(self):
        market_data = MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0})
        executor = RecordingExecutor()
        engine = RealTimeTradingEngine(
            market_data=market_data,
            agent=MockAgent(),
            initial_capital=100_000.0,
            max_leverage=2.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
            execution_mode="paper",
            order_executor=executor,
            commission_rate=0.001,
        )

        result = engine.execute_trading_plan(
            {"actions": [{"symbol": "BTCUSDT", "target_exposure": 1.0}]},
            market_prices={"BTCUSDT": 50_000.0},
            dry_run=True,
        )

        assert result["status"] == "preview"
        assert result["valid"] is True
        assert result["projected_orders"][0]["quantity"] == pytest.approx(2.0)
        assert result["estimated_commission"] == pytest.approx(100.0)
        assert executor.orders == []
        assert engine.trade_log == []
        assert engine.decision_log == []
        assert engine.account.positions == {}

    def test_plan_preview_reports_control_blockers(self):
        engine = RealTimeTradingEngine(
            market_data=MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0}),
            agent=MockAgent(),
            initial_capital=100_000.0,
            max_leverage=2.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )
        engine.read_only_guard = ReadOnlyGuard(enabled=True)

        result = engine.execute_trading_plan(
            {"actions": [{"symbol": "BTCUSDT", "target_exposure": 1.0}]},
            market_prices={"BTCUSDT": 50_000.0},
            dry_run=True,
        )

        assert result["valid"] is False
        assert result["reason"]["code"] == "CONTROL_BLOCKED"
        assert "read_only" in result["control_blockers"]

    def test_rejected_plan_preview_does_not_record_decision(self):
        engine = RealTimeTradingEngine(
            market_data=MockMarketData(["BTCUSDT"], {"BTCUSDT": 50_000.0}),
            agent=MockAgent(),
            initial_capital=100_000.0,
            max_leverage=2.0,
            poll_interval_seconds=5.0,
            decision_interval_seconds=60.0,
        )

        result = engine.execute_trading_plan(
            {"actions": [{"symbol": "UNKNOWN", "target_exposure": 1.0}]},
            market_prices={"BTCUSDT": 50_000.0},
            dry_run=True,
        )

        assert result["status"] == "rejected"
        assert result["reason"]["code"] == "UNKNOWN_SYMBOL"
        assert engine.decision_log == []

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
