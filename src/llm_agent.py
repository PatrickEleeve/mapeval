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
    "- Allocate capital dynamically while targeting strong risk-adjusted returns and controlled drawdowns.\n\n"
    "**Available tools**\n"
    "- get_historical_prices(symbol, end_date, bars): retrieve the latest price history window.\n"
    "- calculate_moving_average(symbol, end_date, window_size): compute moving averages.\n"
    "- calculate_volatility(symbol, end_date, window_size): estimate historical volatility.\n"
    "- calculate_rsi(symbol, end_date, window_size): Relative Strength Index.\n"
    "- calculate_macd(symbol, end_date, fast_period, slow_period, signal_period): MACD suite.\n"
    "- calculate_atr(symbol, end_date, window_size): Average True Range proxy.\n"
    "- calculate_bollinger_bands(symbol, end_date, window_size, num_std): upper/mid/lower bands.\n"
    "- calculate_coefficient_of_variation(symbol, end_date, window_size): std/mean ratio.\n"
    "- calculate_moving_average_slope(symbol, end_date, window_size, periods): trend slope estimate.\n"
    "- get_funding_rate(symbol): recent futures funding rate (per 8h), if available.\n\n"
    "**Exposure scale**\n"
    "- Exposure values are leverage multiples of account equity (not dollar amounts).\n"
    "- Example: 1.0 means a position whose notional equals account equity; 5.0 means controlling five times equity (e.g., "
    "using 100 USDT margin to run a 500 USDT position).\n"
    "- Use conviction-sized exposures (typically 0.25x–{per_symbol_cap:g}x) when you have an edge; stay near 0 only when you "
    "intentionally want to remain neutral.\n\n"
    "**Guidelines**\n"
    "1. Evaluate recent price action and volatility.\n"
    "2. Select leverage exposures for each contract; positive values are long, negative values are short.\n"
    "3. Keep each individual exposure within +/-{per_symbol_cap:g}x.\n"
    "4. Ensure the sum of absolute exposures does not exceed {max_total:g}x total leverage.\n"
    "5. Limit step-to-step changes to +/-{max_delta:g}x per symbol; if you need a larger move, stage it over multiple updates.\n\n"
    "**Output format**\n"
    'Respond strictly in JSON: {{"reasoning": "brief analysis ...", "exposure": {{"BTCUSDT": <float>, "ETHUSDT": <float>}}}}. '
    "Numbers represent leverage multiples of account equity."
)


_SUPPORTED_INDICATORS = {
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "coefficient_of_variation",
    "ma_slope",
}


@dataclass
class LLMAgent:
    """Coordinate prompts and responses with the chosen LLM provider."""

    api_key: str
    config: Dict[str, Any]
    symbols: Optional[List[str]] = None
    max_leverage: float = 50.0
    per_symbol_max_exposure: float = 50.0
    max_exposure_delta: float = 50.0
    min_abs_exposure: float = 0.0
    provider: str = "openai"
    base_url: Optional[str] = None
    indicators: Optional[List[str]] = None
    _system_prompt: str = field(init=False)
    _last_exposures: Dict[str, float] = field(init=False, default_factory=dict)
    last_sanitization_notes: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.symbols = list(self.symbols) if self.symbols else ["BTCUSDT", "ETHUSDT"]
        self.indicators = self._sanitize_indicators(self.indicators)
        self.max_leverage = max(0.0, float(self.max_leverage))
        self.per_symbol_max_exposure = float(self.per_symbol_max_exposure or self.max_leverage)
        if self.max_leverage > 0.0:
            self.per_symbol_max_exposure = min(self.per_symbol_max_exposure, self.max_leverage)
        self.per_symbol_max_exposure = max(0.0, self.per_symbol_max_exposure)
        self.max_exposure_delta = float(self.max_exposure_delta or self.per_symbol_max_exposure or self.max_leverage)
        self.max_exposure_delta = max(0.0, min(self.max_exposure_delta, self.per_symbol_max_exposure or self.max_exposure_delta))
        self.last_reasoning: str = ""
        self.last_indicator_snapshot: str = ""
        self.last_sanitization_notes = []
        self._last_exposures = {symbol: 0.0 for symbol in self.symbols}
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
        return _SYSTEM_PROMPT_TEMPLATE.format(
            symbol_list=symbol_list,
            max_total=self.max_leverage if self.max_leverage > 0 else 0,
            per_symbol_cap=self.per_symbol_max_exposure if self.per_symbol_max_exposure > 0 else 0,
            max_delta=self.max_exposure_delta if self.max_exposure_delta > 0 else 0,
        )

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
        indicator_snapshot = self._format_indicator_snapshot(current_time, available_tools)
        self.last_indicator_snapshot = indicator_snapshot
        user_prompt = (
            f"Current timestamp (UTC): {timestamp_utc}\n"
            f"Tradable contracts: {symbol_hint}\n"
            "Recent closing prices (most recent 60 records):\n"
            f"{formatted_data}\n\n"
            f"Recent funding rates (per 8h): {funding_hint}\n\n"
        )
        if indicator_snapshot:
            user_prompt += f"Technical indicators:\n{indicator_snapshot}\n\n"
        user_prompt += (
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

    @staticmethod
    def _call_tool(tools: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        func = getattr(tools, name, None)
        if not callable(func):
            return None
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    def _sanitize_indicators(self, values: Optional[List[str]]) -> List[str]:
        if not values:
            return []
        normalized: List[str] = []
        seen = set()
        for value in values:
            key = str(value).lower()
            if key not in _SUPPORTED_INDICATORS:
                continue
            if key in seen:
                continue
            normalized.append(key)
            seen.add(key)
        return normalized

    def _format_indicator_snapshot(self, current_time: pd.Timestamp, tools: Any) -> str:
        if not self.indicators:
            return ""
        snapshot_lines: List[str] = []
        for symbol in self.symbols:
            metrics: List[str] = []
            if "rsi" in self.indicators:
                rsi = self._call_tool(tools, "calculate_rsi", symbol, current_time, 14)
                if rsi is not None:
                    metrics.append(f"RSI14={rsi:.2f}")
            if "macd" in self.indicators:
                macd = self._call_tool(tools, "calculate_macd", symbol, current_time)
                if isinstance(macd, dict):
                    metrics.append(
                        "MACD={macd:.4f}|SIG={signal:.4f}|HIST={hist:.4f}".format(
                            macd=macd.get("macd", 0.0),
                            signal=macd.get("signal", 0.0),
                            hist=macd.get("histogram", 0.0),
                        )
                    )
            if "atr" in self.indicators:
                atr = self._call_tool(tools, "calculate_atr", symbol, current_time, 14)
                if atr is not None:
                    metrics.append(f"ATR14={atr:.4f}")
            if "bollinger_bands" in self.indicators:
                bands = self._call_tool(tools, "calculate_bollinger_bands", symbol, current_time, 20, 2.0)
                if isinstance(bands, dict):
                    upper = bands.get("upper")
                    mid = bands.get("mid")
                    lower = bands.get("lower")
                    if None not in (upper, mid, lower):
                        metrics.append(f"BB20=({lower:.2f}/{mid:.2f}/{upper:.2f})")
                    bandwidth = bands.get("bandwidth")
                    if bandwidth is not None:
                        metrics.append(f"BB_bw={bandwidth:.4f}")
            if "coefficient_of_variation" in self.indicators:
                cv = self._call_tool(
                    tools,
                    "calculate_coefficient_of_variation",
                    symbol,
                    current_time,
                    20,
                )
                if cv is not None:
                    metrics.append(f"CV20={cv:.4f}")
            if "ma_slope" in self.indicators:
                slope = self._call_tool(
                    tools,
                    "calculate_moving_average_slope",
                    symbol,
                    current_time,
                    20,
                    5,
                )
                if slope is not None:
                    metrics.append(f"MA20Slope5={slope:.5f}")

            if metrics:
                snapshot_lines.append(f"{symbol}: {', '.join(metrics)}")
        return "\n".join(snapshot_lines)

    def _sanitize_exposures(self, exposures: Dict[str, Any]) -> Optional[Dict[str, float]]:
        self.last_sanitization_notes = []
        sanitized: Dict[str, float] = {}
        per_symbol_cap = max(0.0, self.per_symbol_max_exposure)
        delta_cap = max(0.0, self.max_exposure_delta)
        prev = self._last_exposures or {symbol: 0.0 for symbol in self.symbols}
        for symbol in self.symbols:
            raw = exposures.get(symbol, 0.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                self.last_sanitization_notes.append(f"{symbol}: non-numeric exposure -> fallback")
                return None
            clipped = value
            if per_symbol_cap > 0.0:
                bounded = max(-per_symbol_cap, min(per_symbol_cap, clipped))
                if bounded != clipped:
                    self.last_sanitization_notes.append(
                        f"{symbol}: clipped {clipped:.6f} -> {bounded:.6f} by per-symbol limit ±{per_symbol_cap:.6f}"
                    )
                clipped = bounded
            if delta_cap > 0.0:
                prior = prev.get(symbol, 0.0)
                bounded = max(prior - delta_cap, min(prior + delta_cap, clipped))
                if bounded != clipped:
                    self.last_sanitization_notes.append(
                        f"{symbol}: adjusted {clipped:.6f} -> {bounded:.6f} by delta limit ±{delta_cap:.6f}"
                    )
                clipped = bounded
            sanitized[symbol] = clipped

        if self.max_leverage <= 0.0:
            if any(abs(value) > 1e-12 for value in sanitized.values()):
                self.last_sanitization_notes.append("Max leverage set to 0; zeroing all exposures.")
            sanitized = {symbol: 0.0 for symbol in sanitized}
        else:
            total_abs = sum(abs(value) for value in sanitized.values())
            if total_abs > self.max_leverage + 1e-9 and total_abs > 0.0:
                scale = self.max_leverage / total_abs
                sanitized = {symbol: value * scale for symbol, value in sanitized.items()}
                self.last_sanitization_notes.append(
                    f"Scaled exposures by {scale:.6f} to respect total leverage ±{self.max_leverage:.6f}"
                )

        self._last_exposures = dict(sanitized)
        return {symbol: round(value, 6) for symbol, value in sanitized.items()}

    def _fallback_exposures(
        self,
        current_time: pd.Timestamp,
        available_tools: Any,
        market_data_slice: pd.DataFrame,
    ) -> Dict[str, float]:
        """Produce deterministic exposures when the LLM response is unavailable."""
        raw_exposures: Dict[str, float] = {}
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
            raw_exposures[symbol] = exposure

        sanitized = self._sanitize_exposures(raw_exposures)
        if sanitized is None:
            sanitized = {symbol: 0.0 for symbol in self.symbols}
            self._last_exposures = dict(sanitized)

        self.last_reasoning = (
            "Deterministic fallback exposures derived from price momentum and relative volatility."
        )
        return sanitized


@dataclass
class BaselineAgent:
    """Simple rule-based agent for benchmarking."""

    strategy: str
    symbols: List[str]
    max_leverage: float = 1.0
    last_reasoning: str = ""
    last_sanitization_notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.last_reasoning = f"Baseline strategy: {self.strategy}"

    def generate_trading_signal(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        exposures = {}
        if self.strategy == "buy_hold":
            # 1x total leverage split across symbols
            weight = 1.0 / len(self.symbols) if self.symbols else 0.0
            for sym in self.symbols:
                exposures[sym] = weight

        elif self.strategy == "random":
            import random

            for sym in self.symbols:
                # Random exposure within max leverage
                # Scale so total doesn't exceed max_leverage (roughly)
                val = (random.random() - 0.5) * 2
                exposures[sym] = val * (self.max_leverage / len(self.symbols))

        elif self.strategy == "ma_crossover":
            for sym in self.symbols:
                short_ma = available_tools.calculate_moving_average(sym, current_time, 20)
                long_ma = available_tools.calculate_moving_average(sym, current_time, 50)
                if short_ma is not None and long_ma is not None:
                    if short_ma > long_ma:
                        exposures[sym] = 1.0 * (self.max_leverage / len(self.symbols))
                    else:
                        exposures[sym] = -1.0 * (self.max_leverage / len(self.symbols))
                else:
                    exposures[sym] = 0.0

        return exposures
