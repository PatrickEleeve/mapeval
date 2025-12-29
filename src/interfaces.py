"""Abstract interfaces for dependency injection and extensibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class MarketDataProvider(ABC):
    """Abstract interface for market data sources."""
    
    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """List of tradable symbols."""
        ...
    
    @abstractmethod
    def fetch_latest_prices(self) -> Dict[str, float]:
        """Get the most recent price for each symbol."""
        ...
    
    @abstractmethod
    def latest_prices(self) -> Dict[str, float]:
        """Get cached latest prices without fetching."""
        ...
    
    @abstractmethod
    def append_prices(self, prices: Dict[str, float], timestamp: Optional[pd.Timestamp] = None) -> None:
        """Add new price snapshot to history buffer."""
        ...
    
    @abstractmethod
    def get_recent_window(self, rows: int = 120) -> pd.DataFrame:
        """Get recent price history as DataFrame."""
        ...
    
    @abstractmethod
    def refresh_funding_rates(self, throttle_seconds: int = 60) -> Dict[str, float]:
        """Fetch current funding rates."""
        ...
    
    def start_websocket(self) -> None:
        """Start real-time data stream (optional)."""
        pass
    
    def stop_websocket(self) -> None:
        """Stop real-time data stream (optional)."""
        pass


class TradingAgent(ABC):
    """Abstract interface for trading signal generators."""
    
    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """Symbols this agent can trade."""
        ...
    
    @property
    @abstractmethod
    def last_reasoning(self) -> str:
        """Reasoning from the last signal generation."""
        ...
    
    @property
    @abstractmethod
    def last_sanitization_notes(self) -> List[str]:
        """Notes from exposure sanitization."""
        ...
    
    @abstractmethod
    def generate_trading_signal(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        """Generate target exposures for each symbol."""
        ...


class Reporter(ABC):
    """Abstract interface for session reporting."""
    
    @abstractmethod
    def record_tick(
        self,
        timestamp: pd.Timestamp,
        account: Any,
        prices: Dict[str, float],
    ) -> None:
        """Record a market tick."""
        ...
    
    @abstractmethod
    def record_warning(self, timestamp: pd.Timestamp, message: str) -> None:
        """Record a warning message."""
        ...
    
    @abstractmethod
    def finalize(
        self,
        equity_history: List[Dict[str, Any]],
        trade_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate final report."""
        ...
    
    def start(self) -> None:
        """Start the reporter (optional)."""
        pass
    
    def stop(self) -> None:
        """Stop the reporter (optional)."""
        pass


class RiskChecker(ABC):
    """Abstract interface for risk management."""
    
    @abstractmethod
    def check_order(
        self,
        symbol: str,
        target_exposure: float,
        current_equity: float,
        initial_equity: float,
        current_time: datetime,
    ) -> tuple[bool, List[str]]:
        """Check if an order is allowed by risk limits."""
        ...
    
    @abstractmethod
    def should_force_close(self, current_equity: float, initial_equity: float) -> bool:
        """Check if positions should be forcefully closed."""
        ...
    
    @abstractmethod
    def adjust_exposure_for_risk(
        self,
        target_exposure: float,
        current_equity: float,
        volatility: Optional[float] = None,
    ) -> float:
        """Adjust exposure based on current risk state."""
        ...
    
    @abstractmethod
    def update_equity(self, current_equity: float, timestamp: datetime) -> None:
        """Update risk state with current equity."""
        ...
    
    @abstractmethod
    def record_trade_result(self, pnl: float, timestamp: datetime) -> None:
        """Record the result of a trade for risk tracking."""
        ...


@dataclass
class Order:
    """Represents a trading order."""
    
    symbol: str
    target_exposure: float
    current_exposure: float
    price: float
    timestamp: datetime
    source: str = "agent"
    
    @property
    def delta_exposure(self) -> float:
        return self.target_exposure - self.current_exposure
    
    @property
    def is_increase(self) -> bool:
        return abs(self.target_exposure) > abs(self.current_exposure)
    
    @property
    def is_close(self) -> bool:
        return abs(self.target_exposure) < 1e-8


@dataclass
class TradeResult:
    """Result of an executed trade."""
    
    symbol: str
    action: str
    quantity: float
    price: float
    commission: float
    realized_pnl: float
    timestamp: datetime
    
    @property
    def net_pnl(self) -> float:
        return self.realized_pnl - self.commission


class StrategyManager:
    """Manage multiple trading strategies."""
    
    def __init__(self) -> None:
        self._strategies: Dict[str, tuple[TradingAgent, float]] = {}
    
    def add_strategy(self, name: str, agent: TradingAgent, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        self._strategies[name] = (agent, weight)
    
    def remove_strategy(self, name: str) -> None:
        self._strategies.pop(name, None)
    
    def get_strategies(self) -> Dict[str, tuple[TradingAgent, float]]:
        return dict(self._strategies)
    
    def aggregate_signals(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
        method: str = "weighted_average",
    ) -> Dict[str, float]:
        if not self._strategies:
            return {}
        
        all_signals: List[tuple[Dict[str, float], float]] = []
        
        for name, (agent, weight) in self._strategies.items():
            try:
                signal = agent.generate_trading_signal(
                    current_time, market_data_slice, available_tools
                )
                all_signals.append((signal, weight))
            except Exception:
                continue
        
        if not all_signals:
            return {}
        
        if method == "weighted_average":
            return self._weighted_average(all_signals)
        elif method == "majority_vote":
            return self._majority_vote(all_signals)
        else:
            return self._weighted_average(all_signals)
    
    def _weighted_average(
        self,
        signals: List[tuple[Dict[str, float], float]],
    ) -> Dict[str, float]:
        total_weight = sum(w for _, w in signals)
        if total_weight == 0:
            return {}
        
        all_symbols = set()
        for signal, _ in signals:
            all_symbols.update(signal.keys())
        
        result: Dict[str, float] = {}
        for symbol in all_symbols:
            weighted_sum = 0.0
            for signal, weight in signals:
                weighted_sum += signal.get(symbol, 0.0) * weight
            result[symbol] = weighted_sum / total_weight
        
        return result
    
    def _majority_vote(
        self,
        signals: List[tuple[Dict[str, float], float]],
    ) -> Dict[str, float]:
        all_symbols = set()
        for signal, _ in signals:
            all_symbols.update(signal.keys())
        
        result: Dict[str, float] = {}
        for symbol in all_symbols:
            long_votes = 0.0
            short_votes = 0.0
            neutral_votes = 0.0
            
            for signal, weight in signals:
                exposure = signal.get(symbol, 0.0)
                if exposure > 0.1:
                    long_votes += weight
                elif exposure < -0.1:
                    short_votes += weight
                else:
                    neutral_votes += weight
            
            if long_votes > short_votes and long_votes > neutral_votes:
                avg_long = sum(
                    s.get(symbol, 0.0) * w
                    for s, w in signals
                    if s.get(symbol, 0.0) > 0.1
                ) / max(long_votes, 1e-9)
                result[symbol] = avg_long
            elif short_votes > long_votes and short_votes > neutral_votes:
                avg_short = sum(
                    s.get(symbol, 0.0) * w
                    for s, w in signals
                    if s.get(symbol, 0.0) < -0.1
                ) / max(short_votes, 1e-9)
                result[symbol] = avg_short
            else:
                result[symbol] = 0.0
        
        return result

