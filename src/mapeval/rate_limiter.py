"""Token-bucket rate limiter for Binance API calls.

Tracks request weight usage and enforces rate limits with exponential backoff
when 429 responses or weight limits are approached.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token-bucket rate limiter for REST API calls.

    Binance uses a weight-based system: each endpoint consumes a certain weight,
    and the total weight per minute is capped (default 1200 for spot, 2400 for futures).
    The ``X-MBX-USED-WEIGHT-1M`` response header reports current usage.
    """

    def __init__(
        self,
        max_weight_per_minute: int = 1200,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self._max_weight = max_weight_per_minute
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

        self._used_weight: int = 0
        self._last_reset: float = time.monotonic()
        self._consecutive_429: int = 0
        self._lock = threading.Lock()

    def _maybe_reset(self) -> None:
        """Reset the weight counter if more than 60 seconds have elapsed."""
        now = time.monotonic()
        if now - self._last_reset >= 60.0:
            self._used_weight = 0
            self._last_reset = now
            self._consecutive_429 = 0

    def acquire(self, weight: int = 1) -> None:
        """Block until the request can proceed within rate limits.

        Parameters
        ----------
        weight:
            The weight cost of the upcoming request (default 1).
        """
        with self._lock:
            self._maybe_reset()

            # If we're close to the limit, wait until the window resets
            if self._used_weight + weight > self._max_weight:
                wait = 60.0 - (time.monotonic() - self._last_reset)
                if wait > 0:
                    logger.warning(
                        "Rate limit approaching (%d/%d weight used), waiting %.1fs",
                        self._used_weight,
                        self._max_weight,
                        wait,
                    )
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    self._maybe_reset()

            self._used_weight += weight

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update internal weight tracking from Binance response headers.

        Binance returns ``X-MBX-USED-WEIGHT-1M`` with the actual server-side weight.
        """
        weight_str = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
        if weight_str is not None:
            try:
                with self._lock:
                    self._used_weight = int(weight_str)
            except (ValueError, TypeError):
                pass

    def on_429(self) -> float:
        """Handle a 429 (rate limited) response. Returns seconds to wait."""
        with self._lock:
            self._consecutive_429 += 1
            wait = min(
                self._backoff_base * (2 ** (self._consecutive_429 - 1)),
                self._backoff_max,
            )
        logger.warning(
            "Received 429 rate limit response (attempt %d), backing off %.1fs",
            self._consecutive_429,
            wait,
        )
        return wait

    def on_success(self) -> None:
        """Reset the consecutive 429 counter on a successful response."""
        with self._lock:
            self._consecutive_429 = 0

    @property
    def used_weight(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._used_weight

    @property
    def remaining_weight(self) -> int:
        with self._lock:
            self._maybe_reset()
            return max(0, self._max_weight - self._used_weight)

    def get_status(self) -> Dict[str, Any]:
        """Return current rate limiter state for monitoring."""
        with self._lock:
            self._maybe_reset()
            return {
                "used_weight": self._used_weight,
                "max_weight": self._max_weight,
                "remaining_weight": max(0, self._max_weight - self._used_weight),
                "consecutive_429": self._consecutive_429,
                "seconds_until_reset": max(0.0, 60.0 - (time.monotonic() - self._last_reset)),
            }
