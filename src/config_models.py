"""Configuration models using Pydantic for validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    BaseSettings = object
    Field = lambda *args, **kwargs: None
    field_validator = lambda *args, **kwargs: lambda f: f

import yaml


if PYDANTIC_AVAILABLE:
    class LLMProviderConfig(BaseModel):
        """Configuration for a single LLM provider."""

        model_name: str = Field(default="gpt-4-turbo")
        temperature: float = Field(default=0.2, ge=0.0, le=2.0)
        base_url: Optional[str] = None
        supports_json_response_format: bool = True
        max_tokens: Optional[int] = None
        timeout_seconds: int = Field(default=60, ge=5, le=300)


    class TradingConfig(BaseModel):
        """Trading engine configuration."""

        initial_capital: float = Field(default=100_000.0, ge=0)
        max_leverage: float = Field(default=30.0, ge=1.0, le=125.0)
        per_symbol_max_exposure: float = Field(default=5.0, ge=0.1, le=50.0)
        max_exposure_delta: float = Field(default=2.0, ge=0.1, le=10.0)
        poll_interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
        decision_interval_seconds: float = Field(default=180.0, ge=10.0, le=3600.0)
        taker_fee_rate: float = Field(default=0.0005, ge=0.0, le=0.01)
        slippage: float = Field(default=0.0005, ge=0.0, le=0.01)
        min_long_exposure: float = Field(default=0.0, ge=0.0)
        history_interval: str = Field(default="1m")
        history_lookback: int = Field(default=500, ge=10, le=2000)
        liquidation_threshold: float = Field(default=0.05, ge=0.0, le=0.5)
        gross_leverage_cap: float = Field(default=30.0, ge=0.0, le=125.0)
        net_exposure_cap: float = Field(default=30.0, ge=0.0, le=125.0)
        max_open_positions: int = Field(default=20, ge=1, le=100)
        max_turnover_per_step: float = Field(default=60.0, ge=0.0, le=200.0)

        @field_validator("history_interval")
        @classmethod
        def validate_interval(cls, v: str) -> str:
            valid = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
            if v not in valid:
                raise ValueError(f"Invalid interval: {v}. Must be one of {valid}")
            return v


    class RiskConfig(BaseModel):
        """Risk management configuration."""

        max_drawdown: float = Field(default=0.20, ge=0.01, le=1.0)
        max_daily_loss: float = Field(default=0.05, ge=0.01, le=0.5)
        max_position_duration_hours: float = Field(default=4.0, ge=0.1, le=168.0)
        max_correlation: float = Field(default=0.85, ge=0.0, le=1.0)
        max_single_loss: float = Field(default=0.02, ge=0.001, le=0.1)
        min_equity_floor: float = Field(default=0.10, ge=0.01, le=0.5)
        max_consecutive_losses: int = Field(default=5, ge=1, le=20)
        cooldown_after_loss_seconds: int = Field(default=300, ge=0, le=3600)
        enable_risk_manager: bool = True


    class RateLimitConfig(BaseModel):
        """Rate limiting configuration for exchange API calls."""

        max_requests_per_minute: int = Field(default=1200, ge=10, le=6000)
        max_order_requests_per_minute: int = Field(default=300, ge=10, le=1200)
        backoff_base_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
        backoff_max_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


    class BinanceConfig(BaseModel):
        """Binance exchange connection configuration."""

        use_testnet: bool = Field(default=True)
        rest_base_url: str = Field(default="https://fapi.binance.com")
        testnet_rest_base_url: str = Field(default="https://testnet.binancefuture.com")
        ws_base_url: str = Field(default="wss://fstream.binance.com")
        testnet_ws_base_url: str = Field(default="wss://stream.binancefuture.com")
        recv_window: int = Field(default=5000, ge=1000, le=60000)

        @property
        def active_rest_url(self) -> str:
            return self.testnet_rest_base_url if self.use_testnet else self.rest_base_url

        @property
        def active_ws_url(self) -> str:
            return self.testnet_ws_base_url if self.use_testnet else self.ws_base_url
    
    
    class CacheConfig(BaseModel):
        """Caching configuration."""
        
        enabled: bool = True
        cache_dir: str = ".cache/klines"
        max_age_hours: int = Field(default=24, ge=1, le=168)
        cleanup_on_start: bool = False
    
    
    class LoggingConfig(BaseModel):
        """Logging configuration."""
        
        level: str = Field(default="INFO")
        json_output: bool = False
        log_file: Optional[str] = None
        structured: bool = True
        
        @field_validator("level")
        @classmethod
        def validate_level(cls, v: str) -> str:
            valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            v_upper = v.upper()
            if v_upper not in valid:
                raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
            return v_upper
    
    
    class AppConfig(BaseSettings):
        """Main application configuration."""

        model_config = SettingsConfigDict(
            env_prefix="MAPEVAL_",
            env_nested_delimiter="__",
            case_sensitive=False,
        )

        # Execution mode: simulation (no real orders), paper (real data, simulated fills), live (real orders)
        execution_mode: Literal["simulation", "paper", "live"] = Field(default="simulation")

        symbols: List[str] = Field(default=[
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "TRXUSDT", "DOGEUSDT", "ADAUSDT", "SUIUSDT", "AAVEUSDT",
        ])
        default_provider: str = "openai"
        duration: str = "1h"

        trading: TradingConfig = Field(default_factory=TradingConfig)
        risk: RiskConfig = Field(default_factory=RiskConfig)
        cache: CacheConfig = Field(default_factory=CacheConfig)
        logging: LoggingConfig = Field(default_factory=LoggingConfig)
        rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
        binance: BinanceConfig = Field(default_factory=BinanceConfig)

        providers: Dict[str, LLMProviderConfig] = Field(default_factory=lambda: {
            "openai": LLMProviderConfig(model_name="gpt-4-turbo"),
            "deepseek": LLMProviderConfig(
                model_name="deepseek-chat",
                base_url="https://api.deepseek.com",
            ),
            "qwen": LLMProviderConfig(
                model_name="qwen3-max",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        })

        openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
        deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
        qwen_api_key: Optional[str] = Field(default=None, alias="QWEN_API_KEY")
        binance_api_key: Optional[str] = Field(default=None, alias="BINANCE_API_KEY")
        binance_api_secret: Optional[str] = Field(default=None, alias="BINANCE_API_SECRET")

        def get_api_key(self, provider: str) -> Optional[str]:
            key_map = {
                "openai": self.openai_api_key,
                "deepseek": self.deepseek_api_key,
                "qwen": self.qwen_api_key,
            }
            return key_map.get(provider.lower())

        def get_provider_config(self, provider: str) -> LLMProviderConfig:
            return self.providers.get(provider.lower(), LLMProviderConfig())

        @classmethod
        def from_yaml(cls, path: str | Path) -> "AppConfig":
            path = Path(path)
            if not path.exists():
                return cls()

            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            return cls(**data)

        def to_yaml(self, path: str | Path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = self.model_dump(exclude={
                "openai_api_key", "deepseek_api_key", "qwen_api_key",
                "binance_api_key", "binance_api_secret",
            })

            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        def duration_seconds(self) -> float:
            mapping = {
                "1h": 3600,
                "6h": 6 * 3600,
                "12h": 12 * 3600,
                "1d": 24 * 3600,
            }
            return float(mapping.get(self.duration, 3600))

else:
    class LLMProviderConfig:
        def __init__(self, **kwargs):
            self.model_name = kwargs.get("model_name", "gpt-4-turbo")
            self.temperature = kwargs.get("temperature", 0.2)
            self.base_url = kwargs.get("base_url")
            self.supports_json_response_format = kwargs.get("supports_json_response_format", True)

    class TradingConfig:
        def __init__(self, **kwargs):
            self.initial_capital = kwargs.get("initial_capital", 100_000.0)
            self.max_leverage = kwargs.get("max_leverage", 30.0)
            self.per_symbol_max_exposure = kwargs.get("per_symbol_max_exposure", 5.0)
            self.max_exposure_delta = kwargs.get("max_exposure_delta", 2.0)
            self.poll_interval_seconds = kwargs.get("poll_interval_seconds", 5.0)
            self.decision_interval_seconds = kwargs.get("decision_interval_seconds", 180.0)
            self.taker_fee_rate = kwargs.get("taker_fee_rate", 0.0005)
            self.slippage = kwargs.get("slippage", 0.0005)
            self.history_interval = kwargs.get("history_interval", "1m")
            self.history_lookback = kwargs.get("history_lookback", 500)
            self.liquidation_threshold = kwargs.get("liquidation_threshold", 0.05)
            self.gross_leverage_cap = kwargs.get("gross_leverage_cap", 30.0)
            self.net_exposure_cap = kwargs.get("net_exposure_cap", 30.0)
            self.max_open_positions = kwargs.get("max_open_positions", 20)
            self.max_turnover_per_step = kwargs.get("max_turnover_per_step", 60.0)

    class RiskConfig:
        def __init__(self, **kwargs):
            self.max_drawdown = kwargs.get("max_drawdown", 0.20)
            self.max_daily_loss = kwargs.get("max_daily_loss", 0.05)
            self.max_position_duration_hours = kwargs.get("max_position_duration_hours", 4.0)
            self.max_correlation = kwargs.get("max_correlation", 0.85)
            self.max_single_loss = kwargs.get("max_single_loss", 0.02)
            self.min_equity_floor = kwargs.get("min_equity_floor", 0.10)
            self.max_consecutive_losses = kwargs.get("max_consecutive_losses", 5)
            self.cooldown_after_loss_seconds = kwargs.get("cooldown_after_loss_seconds", 300)
            self.enable_risk_manager = kwargs.get("enable_risk_manager", True)

    class CacheConfig:
        def __init__(self, **kwargs):
            self.enabled = kwargs.get("enabled", True)
            self.cache_dir = kwargs.get("cache_dir", ".cache/klines")
            self.max_age_hours = kwargs.get("max_age_hours", 24)

    class LoggingConfig:
        def __init__(self, **kwargs):
            self.level = kwargs.get("level", "INFO")
            self.json_output = kwargs.get("json_output", False)
            self.structured = kwargs.get("structured", True)

    class RateLimitConfig:
        def __init__(self, **kwargs):
            self.max_requests_per_minute = kwargs.get("max_requests_per_minute", 1200)
            self.max_order_requests_per_minute = kwargs.get("max_order_requests_per_minute", 300)
            self.backoff_base_seconds = kwargs.get("backoff_base_seconds", 1.0)
            self.backoff_max_seconds = kwargs.get("backoff_max_seconds", 60.0)

    class BinanceConfig:
        def __init__(self, **kwargs):
            self.use_testnet = kwargs.get("use_testnet", True)
            self.rest_base_url = kwargs.get("rest_base_url", "https://fapi.binance.com")
            self.testnet_rest_base_url = kwargs.get("testnet_rest_base_url", "https://testnet.binancefuture.com")
            self.ws_base_url = kwargs.get("ws_base_url", "wss://fstream.binance.com")
            self.testnet_ws_base_url = kwargs.get("testnet_ws_base_url", "wss://stream.binancefuture.com")
            self.recv_window = kwargs.get("recv_window", 5000)

        @property
        def active_rest_url(self) -> str:
            return self.testnet_rest_base_url if self.use_testnet else self.rest_base_url

        @property
        def active_ws_url(self) -> str:
            return self.testnet_ws_base_url if self.use_testnet else self.ws_base_url

    class AppConfig:
        def __init__(self, **kwargs):
            self.execution_mode = kwargs.get("execution_mode", "simulation")
            self.symbols = kwargs.get("symbols", ["BTCUSDT", "ETHUSDT"])
            self.default_provider = kwargs.get("default_provider", "openai")
            self.duration = kwargs.get("duration", "1h")
            self.trading = TradingConfig(**kwargs.get("trading", {}))
            self.risk = RiskConfig(**kwargs.get("risk", {}))
            self.cache = CacheConfig(**kwargs.get("cache", {}))
            self.logging = LoggingConfig(**kwargs.get("logging", {}))
            self.rate_limit = RateLimitConfig(**kwargs.get("rate_limit", {}))
            self.binance = BinanceConfig(**kwargs.get("binance", {}))
            self.providers = {}
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            self.qwen_api_key = os.getenv("QWEN_API_KEY")
            self.binance_api_key = os.getenv("BINANCE_API_KEY")
            self.binance_api_secret = os.getenv("BINANCE_API_SECRET")

        def get_api_key(self, provider: str) -> Optional[str]:
            return getattr(self, f"{provider.lower()}_api_key", None)

        def get_provider_config(self, provider: str) -> LLMProviderConfig:
            return self.providers.get(provider.lower(), LLMProviderConfig())

        @classmethod
        def from_yaml(cls, path: str) -> "AppConfig":
            return cls()

        def duration_seconds(self) -> float:
            mapping = {"1h": 3600, "6h": 6*3600, "12h": 12*3600, "1d": 24*3600}
            return float(mapping.get(self.duration, 3600))


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path and Path(config_path).exists():
        return AppConfig.from_yaml(config_path)
    
    default_paths = ["config.yaml", "config.yml", "mapeval.yaml", "mapeval.yml"]
    for path in default_paths:
        if Path(path).exists():
            return AppConfig.from_yaml(path)
    
    return AppConfig()


def generate_default_config(output_path: str = "config.yaml") -> None:
    if PYDANTIC_AVAILABLE:
        config = AppConfig()
        config.to_yaml(output_path)
        print(f"Generated default configuration at {output_path}")
    else:
        print("Pydantic not available. Cannot generate config file.")

