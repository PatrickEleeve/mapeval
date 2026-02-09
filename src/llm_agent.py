"""LLM-powered futures trading agent with structured output and confidence scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

_SYSTEM_PROMPT_TEMPLATE = """You are a professional cryptocurrency perpetual futures portfolio manager.
You may trade: {symbol_list}

**Objective**
- Target strong risk-adjusted returns with controlled drawdowns
- MINIMIZE TURNOVER: Trading costs (0.1% round-trip) compound rapidly. Only trade with CLEAR conviction.
- **HOLD IS YOUR DEFAULT CHOICE.** Most market conditions do not warrant action.

**Trading Philosophy**
- A good trade held for 30+ minutes beats 10 quick trades
- Transaction costs destroy edge: each rebalance costs ~0.1% of turnover
- Patience is alpha. Noise is not signal.
- If current positions are still reasonable, HOLD.

**Exposure scale**
- Values are leverage multiples of equity (1.0x = notional equals equity)
- Use conviction-sized exposures (0.5x-{per_symbol_cap:g}x) ONLY with clear edge
- Flat (0) is valid when uncertain

**HARD CONSTRAINTS (enforced by system)**
1. Per-symbol: +/-{per_symbol_cap:g}x
2. Gross leverage: sum(|w|) <= {gross_leverage_cap:g}x
3. Net exposure: |sum(w)| <= {net_exposure_cap:g}x
4. Max positions: {max_positions:d} symbols
5. Max turnover/step: sum(|Δw|) <= {max_turnover:g}x
6. Step delta: +/-{max_delta:g}x per symbol

**CRITICAL: Pick your BEST {max_positions:d} ideas. You CANNOT hold all symbols.**

**Output format (STRICT JSON)**

OPTION 1 - HOLD (PREFERRED - no changes, keep current positions):
{{
  "action": "HOLD",
  "reasoning": "Why no action is needed right now"
}}

OPTION 2 - REBALANCE (only when truly necessary):
{{
  "action": "REBALANCE",
  "overall_confidence": <0.0-1.0>,
  "reasoning": "1-2 sentence market view",
  "positions": [
    {{
      "symbol": "BTCUSDT",
      "exposure": 1.5,
      "confidence": 0.7,
      "reason": "Oversold RSI + bullish MACD cross"
    }}
  ]
}}

**WHEN TO HOLD (default) vs REBALANCE**
HOLD when:
- No dramatic change in indicators since last decision
- Current positions are still aligned with market direction  
- RSI/MACD signals are ambiguous or unchanged
- Cost of rebalancing > expected benefit

REBALANCE only when:
- Clear reversal signal (RSI crosses 30/70, MACD cross)
- Current positions are WRONG (not just suboptimal)
- New high-conviction opportunity appeared
- Stop-loss level breached

**RULES**
- HOLD is preferred unless you have HIGH CONVICTION (>0.6) for a change
- Rebalancing costs ~0.1% of turnover. A 2x turnover costs 0.2% of equity.
- Only include non-zero exposure symbols in positions array
- Omitted symbols in REBALANCE = 0 exposure (position closed)
- confidence < 0.3 will be rejected
- Empty/vague reasons will be rejected"""

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
    gross_leverage_cap: Optional[float] = None
    net_exposure_cap: Optional[float] = None
    max_open_positions: Optional[int] = None
    max_turnover_per_step: Optional[float] = None
    min_confidence_threshold: float = 0.3
    _system_prompt: str = field(init=False)
    _last_exposures: Dict[str, float] = field(init=False, default_factory=dict)
    last_sanitization_notes: List[str] = field(init=False, default_factory=list)
    last_position_details: List[Dict[str, Any]] = field(init=False, default_factory=list)
    last_overall_confidence: float = field(init=False, default=0.0)
    last_action: str = field(init=False, default="HOLD")

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
        
        if self.gross_leverage_cap is None:
            self.gross_leverage_cap = self.max_leverage
        if self.net_exposure_cap is None:
            self.net_exposure_cap = self.max_leverage
        if self.max_open_positions is None:
            self.max_open_positions = len(self.symbols)
        if self.max_turnover_per_step is None:
            self.max_turnover_per_step = self.max_leverage * 2
        
        self.last_reasoning: str = ""
        self.last_indicator_snapshot: str = ""
        self.last_sanitization_notes = []
        self.last_position_details = []
        self.last_overall_confidence = 0.0
        self.last_action = "HOLD"
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
            per_symbol_cap=self.per_symbol_max_exposure if self.per_symbol_max_exposure > 0 else 0,
            gross_leverage_cap=self.gross_leverage_cap if self.gross_leverage_cap > 0 else 0,
            net_exposure_cap=self.net_exposure_cap if self.net_exposure_cap > 0 else 0,
            max_positions=self.max_open_positions if self.max_open_positions > 0 else len(self.symbols),
            max_turnover=self.max_turnover_per_step if self.max_turnover_per_step > 0 else 0,
            max_delta=self.max_exposure_delta if self.max_exposure_delta > 0 else 0,
        )

    def _format_structured_indicators(self, current_time: pd.Timestamp, tools: Any) -> str:
        if not self.indicators:
            return ""
        
        lines = ["```json", "{"]
        symbol_data = []
        
        for symbol in self.symbols:
            metrics = {}
            
            if "rsi" in self.indicators:
                rsi = self._call_tool(tools, "calculate_rsi", symbol, current_time, 14)
                if rsi is not None:
                    metrics["rsi_14"] = round(rsi, 2)
                    if rsi < 30:
                        metrics["rsi_signal"] = "OVERSOLD"
                    elif rsi > 70:
                        metrics["rsi_signal"] = "OVERBOUGHT"
                    else:
                        metrics["rsi_signal"] = "NEUTRAL"
            
            if "macd" in self.indicators:
                macd = self._call_tool(tools, "calculate_macd", symbol, current_time)
                if isinstance(macd, dict):
                    metrics["macd"] = round(macd.get("macd", 0.0), 6)
                    metrics["macd_signal"] = round(macd.get("signal", 0.0), 6)
                    metrics["macd_hist"] = round(macd.get("histogram", 0.0), 6)
                    if metrics["macd_hist"] > 0 and metrics["macd"] > metrics["macd_signal"]:
                        metrics["macd_trend"] = "BULLISH"
                    elif metrics["macd_hist"] < 0 and metrics["macd"] < metrics["macd_signal"]:
                        metrics["macd_trend"] = "BEARISH"
                    else:
                        metrics["macd_trend"] = "NEUTRAL"
            
            if "atr" in self.indicators:
                atr = self._call_tool(tools, "calculate_atr", symbol, current_time, 14)
                price = self._call_tool(tools, "calculate_moving_average", symbol, current_time, 1)
                if atr is not None and price is not None and price > 0:
                    metrics["atr_14"] = round(atr, 6)
                    metrics["atr_pct"] = round(atr / price * 100, 2)
            
            if "bollinger_bands" in self.indicators:
                bands = self._call_tool(tools, "calculate_bollinger_bands", symbol, current_time, 20, 2.0)
                if isinstance(bands, dict):
                    metrics["bb_upper"] = round(bands.get("upper", 0.0), 4)
                    metrics["bb_mid"] = round(bands.get("mid", 0.0), 4)
                    metrics["bb_lower"] = round(bands.get("lower", 0.0), 4)
                    if bands.get("bandwidth"):
                        metrics["bb_bandwidth"] = round(bands.get("bandwidth", 0.0), 4)
            
            funding = self._call_tool(tools, "get_funding_rate", symbol)
            if funding is not None:
                metrics["funding_rate"] = round(funding, 6)
            
            if metrics:
                symbol_data.append(f'  "{symbol}": {json.dumps(metrics)}')
        
        lines.append(",\n".join(symbol_data))
        lines.append("}")
        lines.append("```")
        
        return "\n".join(lines)

    def generate_trading_signal(
        self,
        current_time: pd.Timestamp,
        market_data_slice: pd.DataFrame,
        available_tools: Any,
    ) -> Dict[str, float]:
        current_time = pd.to_datetime(current_time)
        if isinstance(current_time, pd.Timestamp) and current_time.tzinfo is not None:
            current_time = current_time.tz_convert(None)
        
        timestamp_utc = (
            current_time.tz_localize("UTC") if current_time.tzinfo is None else current_time.tz_convert("UTC")
        )
        
        indicator_json = self._format_structured_indicators(current_time, available_tools)
        self.last_indicator_snapshot = indicator_json
        
        current_positions = []
        for sym, exp in self._last_exposures.items():
            if abs(exp) > 1e-9:
                current_positions.append(f"{sym}:{exp:+.2f}x")
        current_pos_str = ", ".join(current_positions) if current_positions else "FLAT (no positions)"
        
        user_prompt = f"""Current timestamp (UTC): {timestamp_utc}

**Current positions:** {current_pos_str}

**Technical indicators (structured):**
{indicator_json}

**Your task:**
1. Analyze the indicators above
2. Decide: HOLD (keep current positions) or REBALANCE (adjust positions)
3. If REBALANCE: select up to {self.max_open_positions} best opportunities with exposure and confidence
4. Remember: trading costs ~0.1% of turnover. Only rebalance with HIGH CONVICTION (>0.6).

Return your decision as JSON (HOLD or REBALANCE format)."""

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
                exposures = self._parse_structured_response(parsed)
                if exposures is not None:
                    return exposures
            except Exception as e:
                self.last_reasoning = f"Falling back to heuristic: {str(e)[:100]}"

        return self._fallback_exposures(current_time, available_tools, market_data_slice)

    def _parse_structured_response(self, parsed: Dict[str, Any]) -> Optional[Dict[str, float]]:
        self.last_sanitization_notes = []
        self.last_position_details = []
        
        action = str(parsed.get("action", "REBALANCE")).upper()
        self.last_action = action
        self.last_reasoning = str(parsed.get("reasoning", ""))
        
        if action == "HOLD":
            self.last_sanitization_notes.append("LLM chose HOLD - keeping current positions")
            self.last_overall_confidence = 0.0
            return dict(self._last_exposures)
        
        self.last_overall_confidence = float(parsed.get("overall_confidence", 0.0))
        
        if self.last_overall_confidence < self.min_confidence_threshold:
            self.last_sanitization_notes.append(
                f"Overall confidence {self.last_overall_confidence:.2f} < {self.min_confidence_threshold:.2f}, keeping positions"
            )
            return dict(self._last_exposures)
        
        positions = parsed.get("positions", [])
        if isinstance(positions, dict):
            positions = [{"symbol": k, "exposure": v} for k, v in positions.items()]
        
        if not positions:
            self.last_sanitization_notes.append("No positions in REBALANCE response, going flat")
            exposures = {symbol: 0.0 for symbol in self.symbols}
            self._last_exposures = dict(exposures)
            return exposures
        
        exposures: Dict[str, float] = {symbol: 0.0 for symbol in self.symbols}
        
        for pos in positions:
            symbol = str(pos.get("symbol", "")).upper()
            if symbol not in self.symbols:
                self.last_sanitization_notes.append(f"Unknown symbol {symbol}, skipped")
                continue
            
            try:
                exposure = float(pos.get("exposure", 0.0))
            except (TypeError, ValueError):
                self.last_sanitization_notes.append(f"{symbol}: invalid exposure value")
                continue
            
            confidence = float(pos.get("confidence", 0.5))
            reason = str(pos.get("reason", ""))
            
            if confidence < self.min_confidence_threshold:
                self.last_sanitization_notes.append(
                    f"{symbol}: confidence {confidence:.2f} < {self.min_confidence_threshold:.2f}, zeroed"
                )
                continue
            
            if len(reason.strip()) < 10:
                self.last_sanitization_notes.append(
                    f"{symbol}: reason too short ({len(reason)} chars), scaling down 50%"
                )
                exposure *= 0.5
            
            self.last_position_details.append({
                "symbol": symbol,
                "exposure": exposure,
                "confidence": confidence,
                "reason": reason,
            })
            
            exposures[symbol] = exposure
        
        sanitized = self._sanitize_exposures(exposures)
        return sanitized

    def _sanitize_exposures(self, exposures: Dict[str, Any]) -> Optional[Dict[str, float]]:
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
                if abs(bounded - clipped) > 1e-9:
                    self.last_sanitization_notes.append(
                        f"{symbol}: clipped {clipped:.4f} -> {bounded:.4f} by per-symbol limit ±{per_symbol_cap:.2f}"
                    )
                clipped = bounded
            
            if delta_cap > 0.0:
                prior = prev.get(symbol, 0.0)
                bounded = max(prior - delta_cap, min(prior + delta_cap, clipped))
                if abs(bounded - clipped) > 1e-9:
                    self.last_sanitization_notes.append(
                        f"{symbol}: adjusted {clipped:.4f} -> {bounded:.4f} by delta limit ±{delta_cap:.2f}"
                    )
                clipped = bounded
            
            sanitized[symbol] = clipped

        if self.max_leverage <= 0.0:
            if any(abs(v) > 1e-12 for v in sanitized.values()):
                self.last_sanitization_notes.append("Max leverage 0; zeroing all")
            sanitized = {symbol: 0.0 for symbol in sanitized}
        else:
            total_abs = sum(abs(v) for v in sanitized.values())
            if total_abs > self.max_leverage + 1e-9 and total_abs > 0.0:
                scale = self.max_leverage / total_abs
                sanitized = {k: v * scale for k, v in sanitized.items()}
                self.last_sanitization_notes.append(
                    f"Scaled by {scale:.4f} to respect max leverage {self.max_leverage:.2f}"
                )

        self._last_exposures = dict(sanitized)
        return {symbol: round(value, 6) for symbol, value in sanitized.items()}

    def generate_portfolio_weights(
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
        return self._format_structured_indicators(current_time, tools)

    def _fallback_exposures(
        self,
        current_time: pd.Timestamp,
        available_tools: Any,
        market_data_slice: pd.DataFrame,
    ) -> Dict[str, float]:
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

        self.last_reasoning = "Deterministic fallback from MA momentum + volatility dampening."
        return sanitized


@dataclass
class BaselineAgent:
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
            weight = 1.0 / len(self.symbols) if self.symbols else 0.0
            for sym in self.symbols:
                exposures[sym] = weight

        elif self.strategy == "random":
            import random
            for sym in self.symbols:
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
