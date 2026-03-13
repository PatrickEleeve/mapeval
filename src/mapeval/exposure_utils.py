"""Shared exposure sanitization and fallback logic for LLM agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def sanitize_exposures(
    exposures: Dict[str, Any],
    symbols: List[str],
    per_symbol_max_exposure: float,
    max_exposure_delta: float,
    max_leverage: float,
    last_exposures: Dict[str, float],
    sanitization_notes: List[str],
) -> Optional[Dict[str, float]]:
    """Clip and scale exposures to respect per-symbol, delta, and total leverage limits."""
    sanitized: Dict[str, float] = {}
    per_symbol_cap = max(0.0, per_symbol_max_exposure)
    delta_cap = max(0.0, max_exposure_delta)
    prev = last_exposures or {symbol: 0.0 for symbol in symbols}

    for symbol in symbols:
        raw = exposures.get(symbol, 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            sanitization_notes.append(f"{symbol}: non-numeric exposure -> fallback")
            return None

        clipped = value
        if per_symbol_cap > 0.0:
            bounded = max(-per_symbol_cap, min(per_symbol_cap, clipped))
            if abs(bounded - clipped) > 1e-9:
                sanitization_notes.append(f"{symbol}: clipped {clipped:.4f} -> {bounded:.4f}")
            clipped = bounded

        if delta_cap > 0.0:
            prior = prev.get(symbol, 0.0)
            bounded = max(prior - delta_cap, min(prior + delta_cap, clipped))
            if abs(bounded - clipped) > 1e-9:
                sanitization_notes.append(f"{symbol}: delta limited {clipped:.4f} -> {bounded:.4f}")
            clipped = bounded

        sanitized[symbol] = clipped

    if max_leverage <= 0.0:
        sanitized = {symbol: 0.0 for symbol in sanitized}
    else:
        total_abs = sum(abs(v) for v in sanitized.values())
        if total_abs > max_leverage + 1e-9 and total_abs > 0.0:
            scale = max_leverage / total_abs
            sanitized = {symbol: value * scale for symbol, value in sanitized.items()}
            sanitization_notes.append(f"Scaled by {scale:.4f}")

    return {symbol: round(value, 6) for symbol, value in sanitized.items()}


def compute_fallback_exposures(
    symbols: List[str],
    current_time: pd.Timestamp,
    available_tools: Any,
    max_leverage: float,
) -> Dict[str, float]:
    """Momentum-based heuristic exposures used when the LLM call fails."""
    raw_exposures: Dict[str, float] = {}
    for symbol in symbols:
        short_ma = available_tools.calculate_moving_average(symbol, current_time, 21)
        long_ma = available_tools.calculate_moving_average(symbol, current_time, 63)
        volatility = available_tools.calculate_volatility(symbol, current_time, 63)

        exposure = 0.0
        if short_ma is not None and long_ma is not None and long_ma > 0.0:
            momentum = (short_ma - long_ma) / long_ma
            exposure = momentum * 10.0

        if volatility is not None and volatility > 0.0:
            dampener = max(0.25, min(1.5, 0.02 / volatility))
            exposure *= dampener

        exposure = max(-max_leverage, min(max_leverage, exposure))
        raw_exposures[symbol] = exposure

    return raw_exposures
