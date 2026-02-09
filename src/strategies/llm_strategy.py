"""LLM-based trading strategy.

Wraps the existing LLMAgent to conform to the Strategy interface.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from strategies.base import Strategy, StrategySignal


class LLMStrategy(Strategy):
    """Strategy that uses an LLM to generate trading signals."""

    def __init__(self, agent=None, **kwargs) -> None:
        self._agent = agent
        self._symbols: List[str] = []
        self._config: Dict[str, Any] = kwargs

    @property
    def name(self) -> str:
        return "llm"

    def initialize(self, symbols: List[str], config: Dict[str, Any]) -> None:
        self._symbols = symbols
        self._config.update(config)

    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        market_data: pd.DataFrame,
        tools: Any,
        current_positions: Dict[str, float],
    ) -> StrategySignal:
        if self._agent is None:
            return StrategySignal(
                exposures={s: 0.0 for s in self._symbols},
                action="HOLD",
                reasoning="No LLM agent configured",
            )

        try:
            signal = self._agent.generate_trading_signal(timestamp, market_data, tools)
            return StrategySignal(
                exposures=signal or {},
                action=getattr(self._agent, "last_action", "REBALANCE"),
                reasoning=getattr(self._agent, "last_reasoning", ""),
                confidence=0.5,
            )
        except Exception as exc:
            return StrategySignal(
                exposures={s: 0.0 for s in self._symbols},
                action="HOLD",
                reasoning=f"LLM error: {exc}",
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "llm",
            "provider": self._config.get("provider", "unknown"),
            "model": self._config.get("model_name", "unknown"),
        }
