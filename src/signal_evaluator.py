"""Track and evaluate LLM signal accuracy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import statistics

import pandas as pd


@dataclass
class SignalRecord:
    symbol: str
    timestamp: pd.Timestamp
    predicted_direction: str
    confidence: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    actual_return: Optional[float] = None
    was_correct: Optional[bool] = None
    holding_period_minutes: Optional[float] = None


@dataclass
class SignalEvaluator:
    evaluation_window: int = 50
    min_move_threshold: float = 0.001
    
    pending_signals: Dict[str, SignalRecord] = field(default_factory=dict)
    completed_signals: deque = field(default_factory=lambda: deque(maxlen=500))
    symbol_accuracy: Dict[str, deque] = field(default_factory=dict)
    
    def record_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        self.pending_signals[symbol] = SignalRecord(
            symbol=symbol,
            timestamp=timestamp,
            predicted_direction=direction,
            confidence=confidence,
            entry_price=entry_price,
        )
        
        if symbol not in self.symbol_accuracy:
            self.symbol_accuracy[symbol] = deque(maxlen=self.evaluation_window)
    
    def evaluate_signal(
        self,
        symbol: str,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
    ) -> Optional[SignalRecord]:
        if symbol not in self.pending_signals:
            return None
        
        signal = self.pending_signals.pop(symbol)
        signal.exit_price = exit_price
        signal.exit_timestamp = exit_timestamp
        
        price_change = (exit_price - signal.entry_price) / signal.entry_price
        signal.actual_return = price_change
        
        if signal.predicted_direction == "long":
            signal.was_correct = price_change > self.min_move_threshold
        elif signal.predicted_direction == "short":
            signal.was_correct = price_change < -self.min_move_threshold
        else:
            signal.was_correct = abs(price_change) < self.min_move_threshold
        
        time_diff = exit_timestamp - signal.timestamp
        signal.holding_period_minutes = time_diff.total_seconds() / 60.0
        
        self.completed_signals.append(signal)
        self.symbol_accuracy[symbol].append(1.0 if signal.was_correct else 0.0)
        
        return signal
    
    def get_overall_accuracy(self) -> Optional[float]:
        if not self.completed_signals:
            return None
        
        recent = list(self.completed_signals)[-self.evaluation_window:]
        correct = sum(1 for s in recent if s.was_correct)
        return correct / len(recent)
    
    def get_symbol_accuracy(self, symbol: str) -> Optional[float]:
        if symbol not in self.symbol_accuracy or not self.symbol_accuracy[symbol]:
            return None
        
        return statistics.mean(self.symbol_accuracy[symbol])
    
    def get_confidence_calibration(self) -> Dict[str, Dict]:
        if not self.completed_signals:
            return {}
        
        buckets = {
            "low": {"range": (0.0, 0.4), "signals": [], "accuracy": None},
            "medium": {"range": (0.4, 0.7), "signals": [], "accuracy": None},
            "high": {"range": (0.7, 1.0), "signals": [], "accuracy": None},
        }
        
        for signal in self.completed_signals:
            conf = signal.confidence
            for bucket_name, bucket in buckets.items():
                low, high = bucket["range"]
                if low <= conf < high or (bucket_name == "high" and conf == 1.0):
                    bucket["signals"].append(signal)
                    break
        
        for bucket in buckets.values():
            if bucket["signals"]:
                correct = sum(1 for s in bucket["signals"] if s.was_correct)
                bucket["accuracy"] = correct / len(bucket["signals"])
                bucket["count"] = len(bucket["signals"])
            else:
                bucket["count"] = 0
            del bucket["signals"]
        
        return buckets
    
    def get_best_performing_symbols(self, top_n: int = 5) -> List[Dict]:
        results = []
        for symbol, accuracy_history in self.symbol_accuracy.items():
            if len(accuracy_history) >= 5:
                acc = statistics.mean(accuracy_history)
                results.append({
                    "symbol": symbol,
                    "accuracy": acc,
                    "sample_size": len(accuracy_history),
                })
        
        results.sort(key=lambda x: x["accuracy"], reverse=True)
        return results[:top_n]
    
    def get_worst_performing_symbols(self, top_n: int = 5) -> List[Dict]:
        results = []
        for symbol, accuracy_history in self.symbol_accuracy.items():
            if len(accuracy_history) >= 5:
                acc = statistics.mean(accuracy_history)
                results.append({
                    "symbol": symbol,
                    "accuracy": acc,
                    "sample_size": len(accuracy_history),
                })
        
        results.sort(key=lambda x: x["accuracy"])
        return results[:top_n]
    
    def should_trust_symbol(self, symbol: str, min_accuracy: float = 0.45) -> bool:
        acc = self.get_symbol_accuracy(symbol)
        if acc is None:
            return True
        return acc >= min_accuracy
    
    def get_summary_stats(self) -> Dict:
        if not self.completed_signals:
            return {
                "total_signals": 0,
                "overall_accuracy": None,
                "avg_holding_minutes": None,
                "avg_return": None,
            }
        
        recent = list(self.completed_signals)[-self.evaluation_window:]
        
        returns = [s.actual_return for s in recent if s.actual_return is not None]
        hold_times = [s.holding_period_minutes for s in recent if s.holding_period_minutes]
        
        return {
            "total_signals": len(self.completed_signals),
            "recent_signals": len(recent),
            "overall_accuracy": self.get_overall_accuracy(),
            "avg_holding_minutes": statistics.mean(hold_times) if hold_times else None,
            "avg_return": statistics.mean(returns) if returns else None,
            "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else None,
            "calibration": self.get_confidence_calibration(),
        }
