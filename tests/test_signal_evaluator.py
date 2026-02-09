"""Tests for SignalEvaluator."""

import sys
sys.path.insert(0, "src")

import pytest
import pandas as pd
from signal_evaluator import SignalEvaluator


class TestSignalEvaluator:
    def setup_method(self):
        self.evaluator = SignalEvaluator(evaluation_window=10)
        self.base_time = pd.Timestamp.now()
    
    def test_record_and_evaluate_correct_long(self):
        self.evaluator.record_signal(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.8,
            entry_price=100.0,
            timestamp=self.base_time,
        )
        
        result = self.evaluator.evaluate_signal(
            symbol="BTCUSDT",
            exit_price=102.0,
            exit_timestamp=self.base_time + pd.Timedelta(minutes=30),
        )
        
        assert result is not None
        assert result.was_correct is True
        assert result.actual_return == pytest.approx(0.02, rel=0.01)
    
    def test_record_and_evaluate_wrong_long(self):
        self.evaluator.record_signal(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.8,
            entry_price=100.0,
            timestamp=self.base_time,
        )
        
        result = self.evaluator.evaluate_signal(
            symbol="BTCUSDT",
            exit_price=98.0,
            exit_timestamp=self.base_time + pd.Timedelta(minutes=30),
        )
        
        assert result is not None
        assert result.was_correct is False
    
    def test_record_and_evaluate_correct_short(self):
        self.evaluator.record_signal(
            symbol="ETHUSDT",
            direction="short",
            confidence=0.7,
            entry_price=100.0,
            timestamp=self.base_time,
        )
        
        result = self.evaluator.evaluate_signal(
            symbol="ETHUSDT",
            exit_price=97.0,
            exit_timestamp=self.base_time + pd.Timedelta(minutes=15),
        )
        
        assert result is not None
        assert result.was_correct is True
    
    def test_overall_accuracy(self):
        for i in range(5):
            self.evaluator.record_signal(
                symbol=f"SYM{i}",
                direction="long",
                confidence=0.8,
                entry_price=100.0,
                timestamp=self.base_time,
            )
            exit_price = 102.0 if i < 3 else 98.0
            self.evaluator.evaluate_signal(
                symbol=f"SYM{i}",
                exit_price=exit_price,
                exit_timestamp=self.base_time + pd.Timedelta(minutes=10),
            )
        
        accuracy = self.evaluator.get_overall_accuracy()
        
        assert accuracy == pytest.approx(0.6, rel=0.01)
    
    def test_symbol_accuracy(self):
        for i in range(3):
            self.evaluator.record_signal(
                symbol="BTCUSDT",
                direction="long",
                confidence=0.8,
                entry_price=100.0,
                timestamp=self.base_time + pd.Timedelta(hours=i),
            )
            self.evaluator.evaluate_signal(
                symbol="BTCUSDT",
                exit_price=102.0,
                exit_timestamp=self.base_time + pd.Timedelta(hours=i, minutes=30),
            )
        
        accuracy = self.evaluator.get_symbol_accuracy("BTCUSDT")
        
        assert accuracy == 1.0
    
    def test_should_trust_symbol(self):
        for i in range(5):
            self.evaluator.record_signal(
                symbol="BTCUSDT",
                direction="long",
                confidence=0.8,
                entry_price=100.0,
                timestamp=self.base_time + pd.Timedelta(hours=i),
            )
            exit_price = 102.0 if i < 1 else 98.0
            self.evaluator.evaluate_signal(
                symbol="BTCUSDT",
                exit_price=exit_price,
                exit_timestamp=self.base_time + pd.Timedelta(hours=i, minutes=30),
            )
        
        assert not self.evaluator.should_trust_symbol("BTCUSDT", min_accuracy=0.45)
    
    def test_confidence_calibration(self):
        configs = [
            (0.2, True),
            (0.3, False),
            (0.5, True),
            (0.6, True),
            (0.8, False),
            (0.9, True),
        ]
        
        for i, (conf, correct) in enumerate(configs):
            self.evaluator.record_signal(
                symbol=f"SYM{i}",
                direction="long",
                confidence=conf,
                entry_price=100.0,
                timestamp=self.base_time,
            )
            exit_price = 102.0 if correct else 98.0
            self.evaluator.evaluate_signal(
                symbol=f"SYM{i}",
                exit_price=exit_price,
                exit_timestamp=self.base_time + pd.Timedelta(minutes=10),
            )
        
        calibration = self.evaluator.get_confidence_calibration()
        
        assert "low" in calibration
        assert "medium" in calibration
        assert "high" in calibration
    
    def test_get_summary_stats(self):
        self.evaluator.record_signal(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.8,
            entry_price=100.0,
            timestamp=self.base_time,
        )
        self.evaluator.evaluate_signal(
            symbol="BTCUSDT",
            exit_price=102.0,
            exit_timestamp=self.base_time + pd.Timedelta(minutes=30),
        )
        
        stats = self.evaluator.get_summary_stats()
        
        assert stats["total_signals"] == 1
        assert stats["overall_accuracy"] == 1.0
        assert stats["avg_holding_minutes"] == 30.0
