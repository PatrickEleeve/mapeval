"""Interface module that bridges LLM outputs with the trading engine."""

from __future__ import annotations

from typing import Any, Dict


class TradingModule:
    """Expose a simple method for applying model decisions to the trading engine."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def submit_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a structured trading plan to the underlying trading engine."""
        if not isinstance(plan, dict):
            raise TypeError("Plan payload must be a dictionary.")
        return self.engine.execute_trading_plan(plan)

    def submit_exposures(self, exposure_map: Dict[str, float], reasoning: str = "") -> Dict[str, Any]:
        """Convenience wrapper that builds a trading plan from a pure exposure map."""
        if not isinstance(exposure_map, dict):
            raise TypeError("Exposure map must be a dictionary.")
        actions = []
        for symbol, value in exposure_map.items():
            actions.append({"symbol": symbol, "target_exposure": value})
        plan = {"reasoning": reasoning, "actions": actions}
        return self.submit_plan(plan)
