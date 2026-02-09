"""Base strategy interface for the trading system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StrategySignal:
    """Output from a strategy's signal generation."""

    exposures: Dict[str, float]           # Symbol -> target leverage
    action: str = "REBALANCE"             # HOLD or REBALANCE
    confidence: float = 0.5               # 0-1 confidence level
    reasoning: str = ""                   # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy."""
        ...

    @abstractmethod
    def initialize(self, symbols: List[str], config: Dict[str, Any]) -> None:
        """Initialize the strategy with symbols and configuration."""
        ...

    @abstractmethod
    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        market_data: pd.DataFrame,
        tools: Any,
        current_positions: Dict[str, float],
    ) -> StrategySignal:
        """Generate trading signals based on current market state.

        Parameters
        ----------
        timestamp: Current time
        market_data: DataFrame with OHLCV data for all symbols
        tools: FinancialTools instance with indicator calculations
        current_positions: Current exposure per symbol (leverage multiples)

        Returns
        -------
        StrategySignal with target exposures and metadata
        """
        ...

    def on_trade_result(self, symbol: str, pnl: float, entry_price: float, exit_price: float) -> None:
        """Callback when a trade is completed. Override for learning strategies."""
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters for logging/serialization."""
        return {}
