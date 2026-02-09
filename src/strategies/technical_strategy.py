"""Technical analysis strategies.

Pure rule-based strategies using technical indicators, without LLM involvement.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from strategies.base import Strategy, StrategySignal


class MACrossoverStrategy(Strategy):
    """Moving average crossover strategy.

    Goes long when short MA crosses above long MA, short when it crosses below.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50, exposure_size: float = 1.0, **kwargs) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._exposure_size = exposure_size
        self._symbols: List[str] = []

    @property
    def name(self) -> str:
        return "ma_crossover"

    def initialize(self, symbols: List[str], config: Dict[str, Any]) -> None:
        self._symbols = symbols
        self._short_window = config.get("short_window", self._short_window)
        self._long_window = config.get("long_window", self._long_window)
        self._exposure_size = config.get("exposure_size", self._exposure_size)

    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        market_data: pd.DataFrame,
        tools: Any,
        current_positions: Dict[str, float],
    ) -> StrategySignal:
        exposures = {}
        reasoning_parts = []

        for symbol in self._symbols:
            try:
                ma_short = tools.moving_average(symbol, self._short_window)
                ma_long = tools.moving_average(symbol, self._long_window)

                if ma_short is None or ma_long is None:
                    exposures[symbol] = current_positions.get(symbol, 0.0)
                    continue

                if ma_short > ma_long:
                    exposures[symbol] = self._exposure_size
                    reasoning_parts.append(f"{symbol}: LONG (MA{self._short_window}>{self._long_window})")
                elif ma_short < ma_long:
                    exposures[symbol] = -self._exposure_size
                    reasoning_parts.append(f"{symbol}: SHORT (MA{self._short_window}<{self._long_window})")
                else:
                    exposures[symbol] = 0.0
            except Exception:
                exposures[symbol] = current_positions.get(symbol, 0.0)

        return StrategySignal(
            exposures=exposures,
            action="REBALANCE",
            confidence=0.6,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No MA signals",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "ma_crossover",
            "short_window": self._short_window,
            "long_window": self._long_window,
            "exposure_size": self._exposure_size,
        }


class RSIMeanReversionStrategy(Strategy):
    """RSI mean reversion strategy.

    Goes long when RSI is oversold (<30), short when overbought (>70).
    """

    def __init__(
        self,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exposure_size: float = 1.0,
        **kwargs,
    ) -> None:
        self._oversold = oversold
        self._overbought = overbought
        self._exposure_size = exposure_size
        self._symbols: List[str] = []

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    def initialize(self, symbols: List[str], config: Dict[str, Any]) -> None:
        self._symbols = symbols
        self._oversold = config.get("oversold", self._oversold)
        self._overbought = config.get("overbought", self._overbought)

    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        market_data: pd.DataFrame,
        tools: Any,
        current_positions: Dict[str, float],
    ) -> StrategySignal:
        exposures = {}
        reasoning_parts = []

        for symbol in self._symbols:
            try:
                rsi = tools.rsi(symbol)
                if rsi is None:
                    exposures[symbol] = current_positions.get(symbol, 0.0)
                    continue

                if rsi < self._oversold:
                    exposures[symbol] = self._exposure_size
                    reasoning_parts.append(f"{symbol}: LONG (RSI={rsi:.1f}<{self._oversold})")
                elif rsi > self._overbought:
                    exposures[symbol] = -self._exposure_size
                    reasoning_parts.append(f"{symbol}: SHORT (RSI={rsi:.1f}>{self._overbought})")
                else:
                    exposures[symbol] = 0.0
                    reasoning_parts.append(f"{symbol}: FLAT (RSI={rsi:.1f})")
            except Exception:
                exposures[symbol] = current_positions.get(symbol, 0.0)

        return StrategySignal(
            exposures=exposures,
            action="REBALANCE",
            confidence=0.5,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No RSI signals",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "rsi_mean_reversion",
            "oversold": self._oversold,
            "overbought": self._overbought,
            "exposure_size": self._exposure_size,
        }
