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
    "- Pre-calculated indicators: RSI (14), MACD (12,26,9), MA (20/50) Trends, Volatility (20).\n"
    "- get_funding_rate(symbol): recent futures funding rate (per 8h), if available.\n\n"
    "**Guidelines**\n"
    "1. Evaluate recent price action and volatility.\n"
    "2. For each contract, determine:\n"
    "   - Allocation: Percent of total equity to use (0.0 to 1.0).\n"
    "   - Leverage: Leverage multiplier (1x to 20x or more).\n"
    "   - Side: 'long' or 'short'.\n"
    "3. Aggressively seize opportunities with higher leverage (e.g., 2x-10x) when market conditions permit.\n\n"
    "**Output format**\n"
    'Respond strictly in JSON: {{"reasoning": "brief analysis ...", "positions": {{"BTCUSDT": {{"allocation": 0.5, "leverage": 10, "side": "long"}}, "ETHUSDT": {{"allocation": 0.2, "leverage": 5, "side": "short"}}}}}}. \n'
    "'allocation' is 0.0-1.0 (fraction of equity). 'leverage' is leverage multiplier (e.g. 10). 'side' is 'long' or 'short'."
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
        
        indicator_summaries: List[str] = []
        for symbol in self.symbols:
            parts = []
            try:
                rsi = getattr(available_tools, "calculate_rsi", lambda *a: None)(symbol, current_time)
                macd = getattr(available_tools, "calculate_macd", lambda *a: None)(symbol, current_time)
                ma20 = getattr(available_tools, "calculate_moving_average", lambda *a: None)(symbol, current_time, 20)
                ma50 = getattr(available_tools, "calculate_moving_average", lambda *a: None)(symbol, current_time, 50)
                vol = getattr(available_tools, "calculate_volatility", lambda *a: None)(symbol, current_time, 20)

                if rsi is not None:
                    state = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    parts.append(f"RSI(14)={rsi:.1f} ({state})")
                if macd is not None:
                    hist = macd.get("hist", 0.0)
                    trend = "Bullish" if hist > 0 else "Bearish"
                    parts.append(f"MACD(12,26,9) Hist={hist:.4f} ({trend})")
                if ma20 is not None and ma50 is not None:
                    trend = "Bullish" if ma20 > ma50 else "Bearish"
                    parts.append(f"Trend(MA20/MA50)={trend} (MA20={ma20:.2f}, MA50={ma50:.2f})")
                if vol is not None:
                    parts.append(f"Vol(20)={vol:.4f}")

            except Exception:
                pass
            
            if parts:
                indicator_summaries.append(f"{symbol}: " + ", ".join(parts))
        
        indicator_text = "\n".join(indicator_summaries) if indicator_summaries else "No indicators available."

        symbol_hint = ", ".join(self.symbols)
        timestamp_utc = (
            current_time.tz_localize("UTC") if current_time.tzinfo is None else current_time.tz_convert("UTC")
        )
        user_prompt = (
            f"Current timestamp (UTC): {timestamp_utc}\n"
            f"Tradable contracts: {symbol_hint}\n"
            "Recent closing prices (most recent 60 records):\n"
            f"{formatted_data}\n\n"
            f"Technical Indicators:\n{indicator_text}\n\n"
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
                print(f"\n[LLM RAW RESPONSE]\n{content}\n")
                parsed = json.loads(content)
                
                exposures = {}
                raw_positions = parsed.get("positions", {})
                if raw_positions and isinstance(raw_positions, dict):
                    for sym, data in raw_positions.items():
                        if not isinstance(data, dict):
                            continue
                        try:
                            alloc = float(data.get("allocation", 0.0))
                            lev = float(data.get("leverage", 1.0))
                            side = str(data.get("side", "long")).lower()
                            sign = 1.0 if side == "long" else -1.0
                            exposures[sym] = alloc * lev * sign
                        except Exception:
                            continue
                else:
                    # Fallback to legacy format
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

        max_abs = max((abs(v) for v in sanitized.values()), default=0.0)
        if max_abs > 0 and max_abs < 0.5:
            print(f"[WARNING] LLM output very low leverage (max {max_abs:.4f}x). Auto-scaling by 10x (assuming % intent).")
            sanitized = {k: v * 10.0 for k, v in sanitized.items()}

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
