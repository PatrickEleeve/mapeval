"""Strategy framework for the trading system.

Provides a unified Strategy interface and concrete implementations
for LLM-based and technical analysis strategies.
"""

from strategies.base import Strategy, StrategySignal
from strategies.llm_strategy import LLMStrategy
from strategies.technical_strategy import MACrossoverStrategy, RSIMeanReversionStrategy

STRATEGY_REGISTRY = {
    "llm": LLMStrategy,
    "ma_crossover": MACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create a strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return cls(**kwargs)
