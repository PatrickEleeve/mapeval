"""LLM-powered futures trading agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency during tests
    OpenAI = None  # type: ignore

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional cryptocurrency perpetual futures portfolio manager. "
    "You may trade the following contracts: {symbol_list}.\n\n"
    "**Objective**\n"
    "- Allocate capital dynamically within a leverage band of +/-{leverage}x while targeting strong risk-adjusted returns "
    "and controlled drawdowns.\n\n"
    "**Available tools**\n"
    "- get_historical_prices(symbol, end_date, bars): retrieve the latest price history window.\n"
    "- calculate_moving_average(symbol, end_date, window_size): compute moving averages.\n"
    "- calculate_volatility(symbol, end_date, window_size): estimate historical volatility.\n"
    "- get_funding_rate(symbol): recent futures funding rate (per 8h), if available.\n\n"
    "**Exposure scale**\n"
    "- Exposure values are leverage multiples of account equity (not dollar amounts).\n"
    "- Example: 1.0 means a position whose notional equals account equity; 5.0 means controlling five times equity (e.g., "
    "using 100 USDT margin to run a 500 USDT position).\n"
    "- Use materially sized exposures (typically 0.5x–10x) when you have conviction; return values near 0 only when you "
    "intentionally want to stay neutral.\n\n"
    "**Guidelines**\n"
    "1. Evaluate recent price action and volatility.\n"
    "2. Select leverage exposures for each contract; positive values are long, negative values are short.\n"
    "3. Keep each exposure within +/-{leverage}x and ensure the sum of absolute exposures does not exceed {leverage}.\n\n"
    "**Output format**\n"
    'Respond strictly in JSON: {{"reasoning": "brief analysis ...", "exposure": {{"BTCUSDT": <float>, "ETHUSDT": <float>}}}}. '
    "Numbers represent leverage multiples of account equity."
)


@dataclass
class LLMAgent:
    """Coordinate prompts and responses with the chosen LLM provider."""

    api_key: str
    config: Dict[str, Any]
    symbols: Optional[List[str]] = None
    max_leverage: float = 50.0
    min_abs_exposure: float = 0.0
    provider: str = "openai"
    base_url: Optional[str] = None
    _system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        self.symbols = list(self.symbols) if self.symbols else ["BTCUSDT", "ETHUSDT"]
        self.last_reasoning: str = ""
        self._client: Optional[Any] = None
        self._system_prompt = self._build_system_prompt()

        if OpenAI is None:
            return

        provider = (self.provider or "openai").lower()
        api_key = (self.api_key or "").strip()
        if provider == "openai" and api_key == "YOUR_OPENAI_API_KEY":
            api_key = ""
        if provider == "deepseek" and api_key == "YOUR_DEEPSEEK_API_KEY":
            api_key = ""
        if not api_key:
            return

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if provider == "deepseek":
            client_kwargs["base_url"] = self.base_url or self.config.get("base_url") or "https://api.deepseek.com"
        elif self.base_url:
            client_kwargs["base_url"] = self.base_url

        try:
            self._client = OpenAI(**client_kwargs)
        except Exception:
            self._client = None
        self.provider = provider

    def _build_system_prompt(self) -> str:
        symbol_list = ", ".join(self.symbols)
        leverage = int(self.max_leverage)
        return _SYSTEM_PROMPT_TEMPLATE.format(symbol_list=symbol_list, leverage=leverage)

    def generate_trading_signal(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        """Return leverage exposures for each symbol using the configured LLM."""
        current_time = pd.to_datetime(current_time)
        if isinstance(current_time, pd.Timestamp) and current_time.tzinfo is not None:
            current_time = current_time.tz_convert(None)
        formatted_data = market_data_slice.tail(60).to_string(index=False)
        funding_lines: List[str] = []
        try:
            get_fr = getattr(available_tools, "get_funding_rate", None)
            if callable(get_fr):
                for sym in self.symbols:
                    fr = get_fr(sym)
                    if fr is not None:
                        funding_lines.append(f"{sym}:{fr:.6f}")
        except Exception:
            funding_lines = []
        funding_hint = ", ".join(funding_lines) if funding_lines else "N/A"
        symbol_hint = ", ".join(self.symbols)
        timestamp_utc = (
            current_time.tz_localize("UTC") if current_time.tzinfo is None else current_time.tz_convert("UTC")
        )
        user_prompt = (
            f"Current timestamp (UTC): {timestamp_utc}\n"
            f"Tradable contracts: {symbol_hint}\n"
            "Recent closing prices (most recent 60 records):\n"
            f"{formatted_data}\n\n"
            f"Recent funding rates (per 8h): {funding_hint}\n\n"
            f"Provide target leverage exposures within +/-{self.max_leverage}x."
        )

        if self._client is not None:
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.config.get("model_name", "gpt-4-turbo"),
                    "temperature": self.config.get("temperature", 0.2),
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if self.config.get("supports_json_response_format", False):
                    request_kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content if response.choices else ""
                parsed = json.loads(content)
                exposures = parsed.get("exposure", {})
                self.last_reasoning = str(parsed.get("reasoning", ""))
                sanitized = self._sanitize_exposures(exposures)
                if sanitized is not None:
                    return sanitized
            except Exception:
                self.last_reasoning = "Falling back to heuristic exposures due to LLM error."

        return self._fallback_exposures(current_time, available_tools, market_data_slice)

    def generate_portfolio_weights(  # legacy alias
        self,
        current_date: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        return self.generate_trading_signal(current_date, market_data_slice, available_tools)

    def _sanitize_exposures(self, exposures: Dict[str, Any]) -> Optional[Dict[str, float]]:
        sanitized: Dict[str, float] = {}
        for symbol in self.symbols:
            raw = exposures.get(symbol)
            if raw is None:
                sanitized[symbol] = 0.0
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            value = max(-self.max_leverage, min(self.max_leverage, value))
            sanitized[symbol] = value

        total_abs = sum(abs(value) for value in sanitized.values())
        if total_abs > self.max_leverage and total_abs > 0.0:
            scale = self.max_leverage / total_abs
            sanitized = {symbol: value * scale for symbol, value in sanitized.items()}
        return {symbol: round(value, 6) for symbol, value in sanitized.items()}

    def _fallback_exposures(
        self,
        current_time: pd.Timestamp,
        available_tools: Any,
        market_data_slice: pd.DataFrame,
    ) -> Dict[str, float]:
        """Produce deterministic exposures when the LLM response is unavailable."""
        exposures: Dict[str, float] = {}
        total_abs = 0.0

        for symbol in self.symbols:
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

            exposure = max(-self.max_leverage, min(self.max_leverage, exposure))
            exposures[symbol] = exposure
            total_abs += abs(exposure)

        if total_abs > self.max_leverage and total_abs > 0.0:
            scale = self.max_leverage / total_abs
            exposures = {symbol: round(value * scale, 6) for symbol, value in exposures.items()}
        else:
            exposures = {symbol: round(value, 6) for symbol, value in exposures.items()}

        self.last_reasoning = (
            "Deterministic fallback exposures derived from price momentum and relative volatility."
        )
        return exposures
