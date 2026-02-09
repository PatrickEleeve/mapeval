"""In-process event bus for decoupled component communication.

Supports synchronous and asynchronous event handling with pub/sub pattern.
Components subscribe to specific event types and receive events when published.

Usage::

    bus = EventBus()
    bus.subscribe(EventType.PRICE_UPDATE, on_price_update)
    bus.subscribe(EventType.ORDER_FILLED, on_order_filled)

    bus.publish(price_update({"BTCUSDT": 50000.0}))
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from events import Event, EventType

logger = logging.getLogger(__name__)

# Handler type: sync function that receives an Event
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # coroutine


class EventBus:
    """Thread-safe synchronous event bus with publish/subscribe pattern.

    Handlers are invoked synchronously in the order they were registered.
    Exceptions in handlers are caught and logged to prevent cascade failures.
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._lock = threading.Lock()
        self._event_count: int = 0
        self._error_count: int = 0

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                logger.debug("Subscribed %s to %s", handler.__name__, event_type.value)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler that receives ALL events (for logging, etc.)."""
        with self._lock:
            if handler not in self._global_handlers:
                self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler for a specific event type."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Handlers are called synchronously. Exceptions are caught and logged.
        """
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))
            global_handlers = list(self._global_handlers)
            self._event_count += 1

        for handler in global_handlers + handlers:
            try:
                handler(event)
            except Exception as exc:
                self._error_count += 1
                logger.error(
                    "Error in handler %s for event %s: %s",
                    handler.__name__,
                    event.event_type.value,
                    exc,
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return event bus statistics."""
        with self._lock:
            handler_counts = {
                et.value: len(handlers) for et, handlers in self._handlers.items() if handlers
            }
        return {
            "total_events_published": self._event_count,
            "total_errors": self._error_count,
            "handlers_by_type": handler_counts,
            "global_handlers": len(self._global_handlers),
        }


class AsyncEventBus:
    """Async event bus for use with asyncio-based trading loops.

    Handlers can be either sync or async functions.
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[AsyncEventHandler]] = defaultdict(list)
        self._global_handlers: List[AsyncEventHandler] = []
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._event_count: int = 0

    def subscribe(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: AsyncEventHandler) -> None:
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event, dispatching to handlers asynchronously."""
        self._event_count += 1
        handlers = list(self._handlers.get(event.event_type, []))
        global_handlers = list(self._global_handlers)

        for handler in global_handlers + handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error(
                    "Async handler %s error for %s: %s",
                    handler.__name__,
                    event.event_type.value,
                    exc,
                )

    async def publish_queued(self, event: Event) -> None:
        """Add event to the queue for background processing."""
        await self._queue.put(event)

    async def process_events(self) -> None:
        """Process events from the queue until stopped."""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self.publish(event)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("Event processing error: %s", exc)

    def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
