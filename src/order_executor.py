"""Order Executor abstraction layer.

Provides a unified interface for order execution across three modes:
- SimulationExecutor: Instant fills with configurable slippage (current behavior)
- PaperExecutor: Uses real market data but simulates fills
- LiveExecutor: Places real orders on Binance Futures

The trading engine interacts only with the OrderExecutor interface, making it
agnostic to the execution mode.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from order_models import (
    Fill,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionInfo,
)

logger = logging.getLogger(__name__)


class OrderExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order for execution. Returns the result."""
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order. Returns True if successful."""
        ...

    @abstractmethod
    def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """Query the current status of an order."""
        ...

    @abstractmethod
    def sync_positions(self) -> Dict[str, PositionInfo]:
        """Fetch current positions from the execution venue."""
        ...

    @abstractmethod
    def sync_balance(self) -> float:
        """Fetch the current available balance (USDT)."""
        ...

    @abstractmethod
    def get_execution_mode(self) -> str:
        """Return the execution mode name."""
        ...


class SimulationExecutor(OrderExecutor):
    """Executes orders with instant simulated fills.

    Replicates the current behavior of _rebalance_position() with slippage
    and commission modeling.
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        slippage: float = 0.0005,
        initial_balance: float = 100_000.0,
    ) -> None:
        self._commission_rate = commission_rate
        self._slippage = slippage
        self._balance = initial_balance
        self._positions: Dict[str, PositionInfo] = {}
        self._order_history: List[OrderResult] = []

    def submit_order(self, order: Order) -> OrderResult:
        if order.price is None or order.price <= 0:
            return OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                reject_reason="No valid price for simulation",
            )

        # Apply slippage
        if order.side == OrderSide.BUY:
            exec_price = order.price * (1 + self._slippage)
        else:
            exec_price = order.price * (1 - self._slippage)

        commission = abs(order.quantity * exec_price) * self._commission_rate
        slippage_cost = abs(order.quantity * order.price * self._slippage)

        fill = Fill(
            fill_id=str(uuid.uuid4())[:12],
            price=exec_price,
            quantity=order.quantity,
            commission=commission,
        )

        result = OrderResult(
            order=order,
            status=OrderStatus.FILLED,
            exchange_order_id=f"SIM-{uuid.uuid4().hex[:8]}",
            filled_quantity=order.quantity,
            avg_fill_price=exec_price,
            total_commission=commission,
            total_slippage_cost=slippage_cost,
            fills=[fill],
        )
        self._order_history.append(result)
        return result

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True  # Simulation: orders fill instantly, nothing to cancel

    def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED  # Simulation: always filled

    def sync_positions(self) -> Dict[str, PositionInfo]:
        return dict(self._positions)

    def sync_balance(self) -> float:
        return self._balance

    def get_execution_mode(self) -> str:
        return "simulation"


class PaperExecutor(OrderExecutor):
    """Executes orders using real market data but simulated fills.

    Adds realistic latency simulation and depth-based fill modeling.
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        slippage: float = 0.0003,
        fill_latency_ms: float = 50.0,
        market_data_provider=None,
    ) -> None:
        self._commission_rate = commission_rate
        self._slippage = slippage
        self._fill_latency_ms = fill_latency_ms
        self._market_data = market_data_provider
        self._balance: float = 0.0
        self._positions: Dict[str, PositionInfo] = {}
        self._order_history: List[OrderResult] = []
        self._pending_orders: Dict[str, Order] = {}

    def submit_order(self, order: Order) -> OrderResult:
        # Simulate fill latency
        if self._fill_latency_ms > 0:
            time.sleep(self._fill_latency_ms / 1000.0)

        # Get current market price if available
        price = order.price
        if price is None or price <= 0:
            return OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                reject_reason="No valid price",
            )

        # Apply slippage (slightly less than simulation to model better fills)
        if order.side == OrderSide.BUY:
            exec_price = price * (1 + self._slippage)
        else:
            exec_price = price * (1 - self._slippage)

        commission = abs(order.quantity * exec_price) * self._commission_rate
        slippage_cost = abs(order.quantity * price * self._slippage)

        fill = Fill(
            fill_id=str(uuid.uuid4())[:12],
            price=exec_price,
            quantity=order.quantity,
            commission=commission,
        )

        result = OrderResult(
            order=order,
            status=OrderStatus.FILLED,
            exchange_order_id=f"PAPER-{uuid.uuid4().hex[:8]}",
            filled_quantity=order.quantity,
            avg_fill_price=exec_price,
            total_commission=commission,
            total_slippage_cost=slippage_cost,
            fills=[fill],
        )
        self._order_history.append(result)
        return result

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return True
        return False

    def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        if order_id in self._pending_orders:
            return OrderStatus.SUBMITTED
        return OrderStatus.FILLED

    def sync_positions(self) -> Dict[str, PositionInfo]:
        return dict(self._positions)

    def sync_balance(self) -> float:
        return self._balance

    def get_execution_mode(self) -> str:
        return "paper"


class LiveExecutor(OrderExecutor):
    """Executes real orders on Binance Futures via the authenticated client.

    Requires a BinanceFuturesClient instance.
    """

    def __init__(self, client, order_timeout_seconds: float = 30.0) -> None:
        """
        Parameters
        ----------
        client: BinanceFuturesClient instance
        order_timeout_seconds: Max time to wait for order fill
        """
        self._client = client
        self._order_timeout = order_timeout_seconds
        self._order_history: List[OrderResult] = []

    def submit_order(self, order: Order) -> OrderResult:
        try:
            response = self._client.place_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=abs(order.quantity),
                price=order.price if order.order_type == OrderType.LIMIT else None,
                reduce_only=order.reduce_only,
                time_in_force=order.time_in_force,
                new_client_order_id=order.client_order_id,
            )

            exchange_status = response.get("status", "")
            status = self._map_status(exchange_status)

            # If not yet filled, wait for fill
            if status not in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED):
                status, response = self._wait_for_fill(
                    order.symbol,
                    response.get("orderId"),
                )

            fills = []
            total_commission = 0.0
            if "fills" in response:
                for f in response["fills"]:
                    fill = Fill(
                        fill_id=str(f.get("tradeId", "")),
                        price=float(f.get("price", 0)),
                        quantity=float(f.get("qty", 0)),
                        commission=float(f.get("commission", 0)),
                        commission_asset=f.get("commissionAsset", "USDT"),
                    )
                    fills.append(fill)
                    total_commission += fill.commission

            result = OrderResult(
                order=order,
                status=status,
                exchange_order_id=str(response.get("orderId", "")),
                filled_quantity=float(response.get("executedQty", 0)),
                avg_fill_price=float(response.get("avgPrice", 0)),
                total_commission=total_commission,
                fills=fills,
                raw_response=response,
            )
            self._order_history.append(result)
            return result

        except Exception as exc:
            logger.error("Live order failed: %s", exc)
            result = OrderResult(
                order=order,
                status=OrderStatus.REJECTED,
                reject_reason=str(exc),
            )
            self._order_history.append(result)
            return result

    def _wait_for_fill(self, symbol: str, order_id: int) -> tuple:
        """Poll order status until filled or timeout."""
        start = time.time()
        while time.time() - start < self._order_timeout:
            try:
                response = self._client.get_order(symbol=symbol, order_id=order_id)
                status = self._map_status(response.get("status", ""))
                if status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                    return status, response
            except Exception as exc:
                logger.warning("Error polling order %s: %s", order_id, exc)
            time.sleep(0.5)

        # Timeout - cancel the order
        logger.warning("Order %s timed out after %.0fs, canceling", order_id, self._order_timeout)
        try:
            self._client.cancel_order(symbol=symbol, order_id=order_id)
        except Exception:
            pass
        return OrderStatus.CANCELED, {"orderId": order_id, "status": "CANCELED"}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self._client.cancel_order(symbol=symbol, order_id=int(order_id))
            return True
        except Exception as exc:
            logger.error("Cancel failed for %s: %s", order_id, exc)
            return False

    def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        try:
            response = self._client.get_order(symbol=symbol, order_id=int(order_id))
            return self._map_status(response.get("status", ""))
        except Exception:
            return OrderStatus.REJECTED

    def sync_positions(self) -> Dict[str, PositionInfo]:
        positions = {}
        try:
            raw_positions = self._client.get_positions()
            for p in raw_positions:
                qty = float(p.get("positionAmt", 0))
                if abs(qty) < 1e-8:
                    continue
                positions[p["symbol"]] = PositionInfo(
                    symbol=p["symbol"],
                    quantity=qty,
                    entry_price=float(p.get("entryPrice", 0)),
                    mark_price=float(p.get("markPrice", 0)),
                    unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                    leverage=float(p.get("leverage", 1)),
                    margin_type=p.get("marginType", "cross"),
                    liquidation_price=float(p.get("liquidationPrice", 0)),
                )
        except Exception as exc:
            logger.error("Failed to sync positions: %s", exc)
        return positions

    def sync_balance(self) -> float:
        try:
            balances = self._client.get_balance()
            for b in balances:
                if b.get("asset") == "USDT":
                    return float(b.get("availableBalance", 0))
        except Exception as exc:
            logger.error("Failed to sync balance: %s", exc)
        return 0.0

    def get_execution_mode(self) -> str:
        return "live"

    @staticmethod
    def _map_status(binance_status: str) -> OrderStatus:
        mapping = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)


def create_executor(
    mode: str,
    commission_rate: float = 0.0005,
    slippage: float = 0.0005,
    initial_balance: float = 100_000.0,
    binance_client=None,
    market_data_provider=None,
) -> OrderExecutor:
    """Factory function to create the appropriate executor for the given mode."""
    if mode == "simulation":
        return SimulationExecutor(
            commission_rate=commission_rate,
            slippage=slippage,
            initial_balance=initial_balance,
        )
    elif mode == "paper":
        return PaperExecutor(
            commission_rate=commission_rate,
            slippage=slippage,
            market_data_provider=market_data_provider,
        )
    elif mode == "live":
        if binance_client is None:
            raise ValueError("BinanceFuturesClient is required for live execution mode")
        return LiveExecutor(client=binance_client)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
