"""Event type definitions for the trading system event bus.

Events are the core communication mechanism between decoupled components.
Each event carries a type, timestamp, source, and payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """All event types in the trading system."""

    # Market data events
    PRICE_UPDATE = "PRICE_UPDATE"
    KLINE_UPDATE = "KLINE_UPDATE"
    FUNDING_RATE_UPDATE = "FUNDING_RATE_UPDATE"

    # Signal & decision events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    DECISION_MADE = "DECISION_MADE"

    # Order events
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_CANCELED = "ORDER_CANCELED"
    ORDER_REJECTED = "ORDER_REJECTED"

    # Position events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_CHANGED = "POSITION_CHANGED"

    # Risk events
    RISK_ALERT = "RISK_ALERT"
    STOP_TRIGGERED = "STOP_TRIGGERED"
    MARGIN_WARNING = "MARGIN_WARNING"
    FORCE_CLOSE = "FORCE_CLOSE"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"

    # Session events
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    SHUTDOWN_REQUESTED = "SHUTDOWN_REQUESTED"

    # System events
    ERROR = "ERROR"
    HEALTH_CHECK = "HEALTH_CHECK"
    RECONCILIATION = "RECONCILIATION"


@dataclass
class Event:
    """Base event that flows through the event bus."""

    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: Optional[str] = None

    def __post_init__(self):
        if self.event_id is None:
            import uuid
            self.event_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }


# ── Convenience constructors ────────────────────────────────────────────

def price_update(prices: Dict[str, float], source: str = "websocket") -> Event:
    return Event(
        event_type=EventType.PRICE_UPDATE,
        payload={"prices": prices},
        source=source,
    )


def signal_generated(exposures: Dict[str, float], reasoning: str = "", source: str = "llm_agent") -> Event:
    return Event(
        event_type=EventType.SIGNAL_GENERATED,
        payload={"exposures": exposures, "reasoning": reasoning},
        source=source,
    )


def order_filled(symbol: str, side: str, quantity: float, price: float, commission: float = 0.0) -> Event:
    return Event(
        event_type=EventType.ORDER_FILLED,
        payload={
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "commission": commission,
        },
        source="executor",
    )


def stop_triggered(symbol: str, stop_price: float, current_price: float) -> Event:
    return Event(
        event_type=EventType.STOP_TRIGGERED,
        payload={
            "symbol": symbol,
            "stop_price": stop_price,
            "current_price": current_price,
        },
        source="stop_loss_manager",
    )


def risk_alert(alert_type: str, message: str, severity: str = "warning", details: Optional[Dict] = None) -> Event:
    return Event(
        event_type=EventType.RISK_ALERT,
        payload={
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "details": details or {},
        },
        source="risk_manager",
    )


def margin_warning(equity: float, margin_used: float, utilization: float) -> Event:
    return Event(
        event_type=EventType.MARGIN_WARNING,
        payload={
            "equity": equity,
            "margin_used": margin_used,
            "utilization": utilization,
        },
        source="trading_engine",
    )
