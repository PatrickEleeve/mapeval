"""Async LLM agent for parallel signal generation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from exposure_utils import compute_fallback_exposures, sanitize_exposures

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional cryptocurrency perpetual futures portfolio manager. "
    "You may trade the following contracts: {symbol_list}.\n\n"
    "**Objective**\n"
    "- Allocate capital dynamically while targeting strong risk-adjusted returns and controlled drawdowns.\n\n"
    "**Exposure scale**\n"
    "- Exposure values are leverage multiples of account equity (not dollar amounts).\n"
    "- Example: 1.0 means a position whose notional equals account equity.\n"
    "- Use conviction-sized exposures (typically 0.25x–{per_symbol_cap:g}x) when you have an edge.\n\n"
    "**Guidelines**\n"
    "1. Evaluate recent price action and volatility.\n"
    "2. Select leverage exposures for each contract; positive values are long, negative values are short.\n"
    "3. Keep each individual exposure within +/-{per_symbol_cap:g}x.\n"
    "4. Ensure the sum of absolute exposures does not exceed {max_total:g}x total leverage.\n"
    "5. Limit step-to-step changes to +/-{max_delta:g}x per symbol.\n\n"
    "**Output format**\n"
    'Respond strictly in JSON: {{"reasoning": "brief analysis ...", "exposure": {{"BTCUSDT": <float>, "ETHUSDT": <float>}}}}. '
    "Numbers represent leverage multiples of account equity."
)


@dataclass
class AsyncLLMAgent:
    """Async version of LLMAgent for parallel processing."""
    
    api_key: str
    config: Dict[str, Any]
    symbols: Optional[List[str]] = None
    max_leverage: float = 50.0
    per_symbol_max_exposure: float = 50.0
    max_exposure_delta: float = 50.0
    provider: str = "openai"
    base_url: Optional[str] = None
    _client: Optional[Any] = field(init=False, default=None)
    _system_prompt: str = field(init=False)
    _last_exposures: Dict[str, float] = field(init=False, default_factory=dict)
    last_reasoning: str = field(init=False, default="")
    last_sanitization_notes: List[str] = field(init=False, default_factory=list)
    
    def __post_init__(self) -> None:
        self.symbols = list(self.symbols) if self.symbols else ["BTCUSDT", "ETHUSDT"]
        self.max_leverage = max(0.0, float(self.max_leverage))
        self.per_symbol_max_exposure = float(self.per_symbol_max_exposure or self.max_leverage)
        if self.max_leverage > 0.0:
            self.per_symbol_max_exposure = min(self.per_symbol_max_exposure, self.max_leverage)
        self.max_exposure_delta = float(self.max_exposure_delta or self.per_symbol_max_exposure)
        self._last_exposures = {symbol: 0.0 for symbol in self.symbols}
        self._system_prompt = self._build_system_prompt()
        
        if AsyncOpenAI is None:
            return
        
        api_key = (self.api_key or "").strip()
        if not api_key or api_key.startswith("YOUR_"):
            return
        
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        elif self.provider == "deepseek":
            client_kwargs["base_url"] = self.config.get("base_url", "https://api.deepseek.com")
        
        try:
            self._client = AsyncOpenAI(**client_kwargs)
        except Exception:
            self._client = None
    
    def _build_system_prompt(self) -> str:
        symbol_list = ", ".join(self.symbols)
        return _SYSTEM_PROMPT_TEMPLATE.format(
            symbol_list=symbol_list,
            max_total=self.max_leverage,
            per_symbol_cap=self.per_symbol_max_exposure,
            max_delta=self.max_exposure_delta,
        )
    
    async def generate_trading_signal_async(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
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
            pass
        
        funding_hint = ", ".join(funding_lines) if funding_lines else "N/A"
        timestamp_utc = (
            current_time.tz_localize("UTC") if current_time.tzinfo is None else current_time.tz_convert("UTC")
        )
        
        user_prompt = (
            f"Current timestamp (UTC): {timestamp_utc}\n"
            f"Tradable contracts: {', '.join(self.symbols)}\n"
            "Recent closing prices (most recent 60 records):\n"
            f"{formatted_data}\n\n"
            f"Recent funding rates (per 8h): {funding_hint}\n\n"
            "Provide target leverage exposures while respecting these limits:\n"
            f"- Per symbol: +/-{self.per_symbol_max_exposure:g}x\n"
            f"- Total absolute leverage: {self.max_leverage:g}x\n"
            f"- Max change versus previous step: +/-{self.max_exposure_delta:g}x per symbol\n"
            "Return an exposure for every tracked symbol (0.0 when flat)."
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
                
                response = await self._client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content if response.choices else ""
                parsed = json.loads(content)
                exposures = parsed.get("exposure", {})
                self.last_reasoning = str(parsed.get("reasoning", ""))
                sanitized = self._sanitize_exposures(exposures)
                if sanitized is not None:
                    return sanitized
            except Exception as e:
                self.last_reasoning = f"Async LLM error: {e}"
        
        return await self._fallback_exposures_async(current_time, available_tools)
    
    async def generate_batch_signals_async(
        self,
        symbol_groups: List[List[str]],
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        tasks = []
        for symbols in symbol_groups:
            agent_copy = AsyncLLMAgent(
                api_key=self.api_key,
                config=self.config,
                symbols=symbols,
                max_leverage=self.max_leverage / len(symbol_groups),
                per_symbol_max_exposure=self.per_symbol_max_exposure,
                max_exposure_delta=self.max_exposure_delta,
                provider=self.provider,
                base_url=self.base_url,
            )
            task = agent_copy.generate_trading_signal_async(
                current_time, market_data_slice, available_tools
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined: Dict[str, float] = {}
        for result in results:
            if isinstance(result, dict):
                combined.update(result)
        
        return combined
    
    def _sanitize_exposures(self, exposures: Dict[str, Any]) -> Optional[Dict[str, float]]:
        self.last_sanitization_notes = []
        result = sanitize_exposures(
            exposures=exposures,
            symbols=self.symbols,
            per_symbol_max_exposure=self.per_symbol_max_exposure,
            max_exposure_delta=self.max_exposure_delta,
            max_leverage=self.max_leverage,
            last_exposures=self._last_exposures,
            sanitization_notes=self.last_sanitization_notes,
        )
        if result is not None:
            self._last_exposures = dict(result)
        return result
    
    async def _fallback_exposures_async(
        self,
        current_time: pd.Timestamp,
        available_tools: Any,
    ) -> Dict[str, float]:
        raw_exposures = compute_fallback_exposures(
            symbols=self.symbols,
            current_time=current_time,
            available_tools=available_tools,
            max_leverage=self.max_leverage,
        )

        sanitized = self._sanitize_exposures(raw_exposures)
        if sanitized is None:
            sanitized = {symbol: 0.0 for symbol in self.symbols}
            self._last_exposures = dict(sanitized)

        self.last_reasoning = "Async fallback: momentum-based heuristic"
        return sanitized


async def run_parallel_signals(
    agents: List[AsyncLLMAgent],
    current_time: pd.Timestamp,
    market_data_slice: pd.DataFrame,
    available_tools: Any,
) -> List[Dict[str, float]]:
    tasks = [
        agent.generate_trading_signal_async(current_time, market_data_slice, available_tools)
        for agent in agents
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)

