"""Configuration settings for the real-time MAPEval futures trading project."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None  # type: ignore


def _load_env() -> None:
    """Load environment variables from a ``.env`` file when available."""
    if load_dotenv is None:
        return
    default_path = Path(__file__).resolve().parent / ".env"
    dotenv_path = os.getenv("MAPEVAL_ENV_FILE")
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv(default_path)


_load_env()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

AGENT_CONFIG = {
    "openai": {
        "model_name": "gpt-5",
        "temperature": 1,
        "supports_json_response_format": True,
    },
    "deepseek": {
        "model_name": "deepseek-chat",
        "temperature": 1,
        "base_url": "https://api.deepseek.com",
        "supports_json_response_format": True,
    },
    "qwen": {
        "model_name": "qwen3-max",
        "temperature": 1,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "supports_json_response_format": True,
    },
}

TRADING_CONFIG = {
    "symbols": [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "TRXUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "SUIUSDT",
        "AAVEUSDT",
        "LINKUSDT",
        "MATICUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "OPUSDT",
        "ARBUSDT",
        "NEARUSDT",
        "ATOMUSDT",
        "LTCUSDT",
        "FTMUSDT",
    ],
    "initial_capital": 100_000.0,
    "poll_interval_seconds": 5.0,
    "decision_interval_seconds": 180.0,
    "history_interval": "1m",
    "history_lookback": 500,
    "max_leverage": 30,
    "per_symbol_max_exposure": 30.0,
    "max_exposure_delta": 30.0,
    "commission_rate": 0.0005,
    "slippage": 0.0005,
    "liquidation_threshold": 0.05,
    "gross_leverage_cap": 30.0,
    "net_exposure_cap": 30.0,
    "max_open_positions": 20,
    "max_turnover_per_step": 60.0,
    "duration_seconds": {
        "1h": 60 * 60,
        "6h": 6 * 60 * 60,
        "12h": 12 * 60 * 60,
        "1d": 24 * 60 * 60,
    },
}
