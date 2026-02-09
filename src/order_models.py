"""Data models for the Order Management System (OMS).

Defines order lifecycle states, order types, fills, and results that are
shared between all executor implementations (simulation, paper, live).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"           # Created but not yet submitted
    SUBMITTED = "SUBMITTED"       # Sent to exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None             # For LIMIT orders
    stop_price: Optional[float] = None        # For STOP orders
    reduce_only: bool = False
    time_in_force: Optional[str] = None       # GTC, IOC, FOK
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def notional(self) -> float:
        price = self.price or 0.0
        return abs(self.quantity * price)


@dataclass
class Fill:
    """A single fill (partial or complete) for an order."""

    fill_id: str
    price: float
    quantity: float
    commission: float
    commission_asset: str = "USDT"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional(self) -> float:
        return abs(self.price * self.quantity)


@dataclass
class OrderResult:
    """Result of submitting an order to an executor."""

    order: Order
    status: OrderStatus
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    total_slippage_cost: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    reject_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED

    @property
    def realized_pnl(self) -> float:
        """Net P&L from this order (fills minus commissions)."""
        return -self.total_commission  # Actual P&L tracked at position level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.order.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.order.symbol,
            "side": self.order.side.value,
            "type": self.order.order_type.value,
            "quantity": self.order.quantity,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "total_commission": self.total_commission,
            "total_slippage_cost": self.total_slippage_cost,
            "reject_reason": self.reject_reason,
            "timestamp": self.timestamp.isoformat(),
            "num_fills": len(self.fills),
        }


@dataclass
class PositionInfo:
    """Snapshot of a position from the exchange or simulation."""

    symbol: str
    quantity: float              # Positive = long, negative = short
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    margin_type: str = "cross"   # "cross" or "isolated"
    liquidation_price: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.mark_price)

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "LONG"
        elif self.quantity < 0:
            return "SHORT"
        return "FLAT"
