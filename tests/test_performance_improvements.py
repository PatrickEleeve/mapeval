"""Tests for performance improvements: caching, single-pass iterations, and efficient slicing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from performance_analyzer import PerformanceAnalyzer
from log_manager import SessionLogger
from factor_evaluator import FactorEvaluator
from data_manager import BacktestMarketData


# ---------------------------------------------------------------------------
# performance_analyzer: calculate_returns() called only once in get_full_report
# ---------------------------------------------------------------------------

class TestPerformanceAnalyzerCachedReturns:
    """Verify that optional `returns` parameter avoids redundant iteration."""

    def _make_analyzer(self) -> PerformanceAnalyzer:
        pa = PerformanceAnalyzer()
        ts = pd.Timestamp("2024-01-01")
        for i, equity in enumerate([100_000, 101_000, 99_000, 102_000, 103_000]):
            pa.record_equity(ts + pd.Timedelta(hours=i), float(equity))
        return pa

    def test_sharpe_accepts_precomputed_returns(self):
        pa = self._make_analyzer()
        returns = pa.calculate_returns()
        sharpe_cached = pa.calculate_sharpe_ratio(returns)
        sharpe_direct = pa.calculate_sharpe_ratio()
        assert sharpe_cached == pytest.approx(sharpe_direct)

    def test_sortino_accepts_precomputed_returns(self):
        pa = self._make_analyzer()
        returns = pa.calculate_returns()
        sortino_cached = pa.calculate_sortino_ratio(returns)
        sortino_direct = pa.calculate_sortino_ratio()
        assert sortino_cached == pytest.approx(sortino_direct)

    def test_get_full_report_uses_returns_once(self):
        """get_full_report should produce identical risk metrics whether returns
        are pre-computed once or computed inside each ratio method."""
        pa = self._make_analyzer()
        report = pa.get_full_report()
        risk = report["risk_metrics"]
        # Both Sharpe and Sortino should be present and consistent
        assert risk["sharpe_ratio"] == pytest.approx(pa.calculate_sharpe_ratio())
        assert risk["sortino_ratio"] == pytest.approx(pa.calculate_sortino_ratio())

    def test_sharpe_with_empty_returns_returns_none(self):
        pa = PerformanceAnalyzer()
        assert pa.calculate_sharpe_ratio([]) is None

    def test_sortino_with_no_downside_returns_none(self):
        # All positive returns → no downside → None
        pa = PerformanceAnalyzer()
        returns = [0.01, 0.02, 0.03]
        assert pa.calculate_sortino_ratio(returns) is None


# ---------------------------------------------------------------------------
# log_manager: shallow copy is sufficient (no deep-copy of nested lists)
# ---------------------------------------------------------------------------

class TestLogManagerShallowCopy:
    def test_save_session_does_not_mutate_original_summary(self, tmp_path):
        logger = SessionLogger(log_dir=tmp_path)
        decision_log = [
            {"timestamp": "2024-01-01T00:00:00Z", "reasoning": "buy signal", "action": "REBALANCE"},
            {"timestamp": "2024-01-01T01:00:00Z", "reasoning": "", "action": "HOLD"},
        ]
        summary = {
            "equity_history": [{"t": 0, "equity": 100_000}],
            "trade_log": [],
            "decision_log": decision_log,
        }
        original_keys = set(summary.keys())
        logger.save_session(run_args={"llm_provider": "test"}, summary=summary, start_time="2024-01-01T00:00:00Z")
        # Shallow copy means original summary should not have 'llm_reasoning' added
        assert set(summary.keys()) == original_keys

    def test_save_session_writes_reasoning_to_file(self, tmp_path):
        logger = SessionLogger(log_dir=tmp_path)
        decision_log = [
            {"timestamp": "2024-01-01T00:00:00Z", "reasoning": "upward trend", "action": "REBALANCE"},
        ]
        summary = {"decision_log": decision_log}
        logger.save_session(run_args={}, summary=summary, start_time="2024-01-01T00:00:00Z")
        jsonl_files = list(tmp_path.glob("*_llm_decisions.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text()
        assert "upward trend" in content


# ---------------------------------------------------------------------------
# factor_evaluator: single-pass trade_log produces correct counts
# ---------------------------------------------------------------------------

class TestFactorEvaluatorSinglePass:
    def _make_trade_log(self):
        return [
            {"action": "open", "realized_pnl": 0},
            {"action": "close", "realized_pnl": 200},
            {"action": "close", "realized_pnl": -100},
            {"action": "reduce", "realized_pnl": 50},
            {"action": "reverse_close", "realized_pnl": -30},
            {"action": "close", "realized_pnl": 0},
        ]

    def test_direction_accuracy_counts_winning_close_actions(self):
        evaluator = FactorEvaluator()
        trade_log = self._make_trade_log()
        session_data = {
            "metadata": {"session_id": "test", "llm_provider": "test"},
            "parameters": {},
            "summary": {
                "equity_history": [{"equity": 100_000}, {"equity": 101_000}],
                "trade_log": trade_log,
                "decision_log": [],
            },
        }
        result = evaluator.evaluate_session(session_data)
        signal_factor = next(f for f in result.factors if f.name == "signal")
        direction_sf = next(sf for sf in signal_factor.sub_factors if sf.name == "direction_accuracy")
        # close_actions: close(+200), close(-100), reduce(+50), reverse_close(-30), close(0) → 5 total
        # winning (pnl>0): 200 and 50 → 2 wins
        assert direction_sf.details["total"] == 5
        assert direction_sf.details["winning"] == 2

    def test_direction_accuracy_ignores_open_actions(self):
        evaluator = FactorEvaluator()
        # Only "open" actions - no close actions
        trade_log = [{"action": "open", "realized_pnl": 500}]
        session_data = {
            "metadata": {"session_id": "test", "llm_provider": "test"},
            "parameters": {},
            "summary": {
                "equity_history": [{"equity": 100_000}, {"equity": 101_000}],
                "trade_log": trade_log,
                "decision_log": [],
            },
        }
        result = evaluator.evaluate_session(session_data)
        signal_factor = next(f for f in result.factors if f.name == "signal")
        direction_sf = next(sf for sf in signal_factor.sub_factors if sf.name == "direction_accuracy")
        assert direction_sf.details["total"] == 0
        # Default accuracy of 50 when no close trades
        assert direction_sf.score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# data_manager: BacktestMarketData.append_prices uses O(1) slice
# ---------------------------------------------------------------------------

class TestBacktestMarketDataSlice:
    def _make_history(self, n: int = 10) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        return pd.DataFrame({"BTCUSDT_Close": [float(100 + i) for i in range(n)]}, index=idx)

    def test_append_prices_advances_window(self):
        history = self._make_history(20)
        mdata = BacktestMarketData(history, symbols=["BTCUSDT"], lookback=5)
        initial_idx = mdata.current_idx
        mdata.append_prices({})
        assert mdata.current_idx == initial_idx + 1

    def test_price_history_length_capped_at_lookback(self):
        history = self._make_history(20)
        lookback = 5
        mdata = BacktestMarketData(history, symbols=["BTCUSDT"], lookback=lookback)
        for _ in range(10):
            mdata.append_prices({})
        assert len(mdata.price_history) <= lookback

    def test_latest_prices_returns_correct_value(self):
        history = self._make_history(10)
        mdata = BacktestMarketData(history, symbols=["BTCUSDT"], lookback=5)
        mdata.append_prices({})
        prices = mdata.latest_prices()
        # After first append: current_idx starts at 5 (lookback), end_idx=6, start_idx=1
        # price_history = full_history.iloc[1:6], so last row is index 5 → Close = 100 + 5 = 105
        assert "BTCUSDT" in prices
        assert prices["BTCUSDT"] == pytest.approx(105.0)

    def test_stops_at_end_of_history(self):
        history = self._make_history(3)
        mdata = BacktestMarketData(history, symbols=["BTCUSDT"], lookback=5)
        for _ in range(10):  # more calls than available data
            mdata.append_prices({})
        assert mdata.current_idx == len(history)
