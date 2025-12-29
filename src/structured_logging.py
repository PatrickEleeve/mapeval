"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


def add_timestamp(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return event_dict


def add_service_info(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    event_dict["service"] = "mapeval"
    event_dict["version"] = "0.2.0"
    return event_dict


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> Any:
    if not STRUCTLOG_AVAILABLE:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        return logging.getLogger("mapeval")
    
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_timestamp,
        add_service_info,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.getLogger().addHandler(file_handler)
    
    return structlog.get_logger()


def get_logger(name: Optional[str] = None) -> Any:
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name) if name else structlog.get_logger()
    return logging.getLogger(name or "mapeval")


class TradingLogger:
    """High-level logging interface for trading operations."""
    
    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("trading")
    
    def log_signal(
        self,
        symbol: str,
        exposure: float,
        reasoning: str,
        latency_ms: float,
        provider: str,
    ) -> None:
        self.logger.info(
            "signal_generated",
            symbol=symbol,
            exposure=exposure,
            reasoning=reasoning[:200] if reasoning else "",
            latency_ms=round(latency_ms, 2),
            provider=provider,
        )
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        commission: float,
        realized_pnl: float,
    ) -> None:
        self.logger.info(
            "trade_executed",
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            commission=commission,
            realized_pnl=realized_pnl,
        )
    
    def log_position_update(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        mark_price: float,
        unrealized_pnl: float,
        leverage: float,
    ) -> None:
        self.logger.debug(
            "position_updated",
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            leverage=leverage,
        )
    
    def log_account_update(
        self,
        equity: float,
        balance: float,
        unrealized_pnl: float,
        margin_used: float,
        available_margin: float,
    ) -> None:
        self.logger.info(
            "account_updated",
            equity=equity,
            balance=balance,
            unrealized_pnl=unrealized_pnl,
            margin_used=margin_used,
            available_margin=available_margin,
        )
    
    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "warning",
    ) -> None:
        log_method = getattr(self.logger, severity, self.logger.warning)
        log_method(
            "risk_event",
            event_type=event_type,
            **details,
        )
    
    def log_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.error(
            "error",
            error_type=error_type,
            message=message,
            **(details or {}),
        )
    
    def log_session_start(
        self,
        session_id: str,
        symbols: list,
        provider: str,
        duration_seconds: float,
        max_leverage: float,
    ) -> None:
        self.logger.info(
            "session_started",
            session_id=session_id,
            symbols=symbols,
            provider=provider,
            duration_seconds=duration_seconds,
            max_leverage=max_leverage,
        )
    
    def log_session_end(
        self,
        session_id: str,
        final_equity: float,
        total_return: float,
        trade_count: int,
        duration_seconds: float,
    ) -> None:
        self.logger.info(
            "session_ended",
            session_id=session_id,
            final_equity=final_equity,
            total_return=total_return,
            trade_count=trade_count,
            duration_seconds=duration_seconds,
        )

