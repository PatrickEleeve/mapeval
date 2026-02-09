"""Adaptive decision interval based on market volatility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import statistics

import pandas as pd


@dataclass
class AdaptiveIntervalManager:
    base_interval_seconds: float = 180.0
    min_interval_seconds: float = 60.0
    max_interval_seconds: float = 600.0
    
    volatility_window: int = 20
    volatility_threshold_low: float = 0.001
    volatility_threshold_high: float = 0.005
    
    price_history: Dict[str, deque] = field(default_factory=dict)
    current_interval: float = field(init=False)
    last_adjustment_reason: str = ""
    
    def __post_init__(self) -> None:
        self.current_interval = self.base_interval_seconds
    
    def update_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.volatility_window)
            self.price_history[symbol].append({"price": price, "timestamp": timestamp})
    
    def calculate_portfolio_volatility(self) -> Optional[float]:
        volatilities = []
        
        for symbol, history in self.price_history.items():
            if len(history) < 5:
                continue
            
            prices = [h["price"] for h in history]
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if len(returns) >= 3:
                vol = statistics.stdev(returns) if len(returns) > 1 else 0.0
                volatilities.append(vol)
        
        if not volatilities:
            return None
        
        return statistics.mean(volatilities)
    
    def get_recommended_interval(self) -> float:
        volatility = self.calculate_portfolio_volatility()
        
        if volatility is None:
            self.last_adjustment_reason = "insufficient data, using base interval"
            return self.base_interval_seconds
        
        if volatility > self.volatility_threshold_high:
            new_interval = self.min_interval_seconds
            self.last_adjustment_reason = f"high volatility ({volatility:.4f}), using minimum interval"
        elif volatility < self.volatility_threshold_low:
            new_interval = self.max_interval_seconds
            self.last_adjustment_reason = f"low volatility ({volatility:.4f}), using maximum interval"
        else:
            ratio = (volatility - self.volatility_threshold_low) / (
                self.volatility_threshold_high - self.volatility_threshold_low
            )
            new_interval = self.max_interval_seconds - ratio * (
                self.max_interval_seconds - self.min_interval_seconds
            )
            self.last_adjustment_reason = f"moderate volatility ({volatility:.4f}), scaling interval"
        
        new_interval = max(self.min_interval_seconds, min(self.max_interval_seconds, new_interval))
        
        smoothing = 0.3
        self.current_interval = (
            smoothing * new_interval + (1 - smoothing) * self.current_interval
        )
        
        return self.current_interval
    
    def detect_regime_change(self) -> Optional[str]:
        volatility = self.calculate_portfolio_volatility()
        
        if volatility is None:
            return None
        
        if volatility > self.volatility_threshold_high * 1.5:
            return "EXTREME_VOLATILITY"
        elif volatility > self.volatility_threshold_high:
            return "HIGH_VOLATILITY"
        elif volatility < self.volatility_threshold_low * 0.5:
            return "VERY_LOW_VOLATILITY"
        elif volatility < self.volatility_threshold_low:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL"
    
    def should_force_decision(self, prices: Dict[str, float]) -> bool:
        if not self.price_history:
            return False
        
        for symbol, current_price in prices.items():
            if symbol not in self.price_history or not self.price_history[symbol]:
                continue
            
            last_price = self.price_history[symbol][-1]["price"]
            if last_price > 0:
                change = abs(current_price - last_price) / last_price
                if change > 0.02:
                    return True
        
        return False
    
    def get_stats(self) -> Dict:
        return {
            "current_interval_seconds": round(self.current_interval, 1),
            "portfolio_volatility": self.calculate_portfolio_volatility(),
            "regime": self.detect_regime_change(),
            "last_adjustment_reason": self.last_adjustment_reason,
            "symbols_tracked": len(self.price_history),
        }
