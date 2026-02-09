"""Security utilities for the trading system.

Provides API key management, audit logging for live trades, and
read-only mode enforcement.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Immutable audit log for live trading operations.

    Every order placed in live mode is logged with full request/response
    details, signed with a hash for tamper detection.
    """

    def __init__(self, log_dir: str = "logs/audit") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._log_file = self._log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log_order(
        self,
        action: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_id: Optional[str],
        response: Optional[Dict[str, Any]] = None,
        execution_mode: str = "live",
    ) -> None:
        """Log an order action with full details."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "execution_mode": execution_mode,
            "response_summary": self._sanitize_response(response) if response else None,
        }

        # Add integrity hash
        entry_json = json.dumps(entry, sort_keys=True)
        entry["integrity_hash"] = hashlib.sha256(entry_json.encode()).hexdigest()[:16]

        self._entries.append(entry)

        # Append to file immediately
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("Failed to write audit log: %s", exc)

    def _sanitize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from response before logging."""
        sanitized = dict(response)
        for key in ("signature", "apiKey", "secretKey"):
            sanitized.pop(key, None)
        return sanitized

    def get_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    @property
    def log_file(self) -> Path:
        return self._log_file


class ReadOnlyGuard:
    """Prevents order placement when read-only mode is enabled.

    Acts as a wrapper around an order executor to block mutations.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def is_read_only(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        logger.warning("Read-only mode ENABLED - all order placement blocked")

    def disable(self) -> None:
        self._enabled = False
        logger.info("Read-only mode disabled")

    def check(self, operation: str = "order") -> None:
        """Raise if read-only mode is active."""
        if self._enabled:
            raise PermissionError(
                f"Operation '{operation}' blocked: system is in read-only mode"
            )


def mask_key(key: str) -> str:
    """Mask an API key for safe display (show first 4 and last 4 chars)."""
    if not key or len(key) < 12:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def validate_api_keys(api_key: Optional[str], api_secret: Optional[str], exchange: str = "binance") -> List[str]:
    """Validate that API keys are present and properly formatted.

    Returns a list of warning messages (empty if all good).
    """
    warnings = []

    if not api_key:
        warnings.append(f"{exchange} API key is not set")
    elif len(api_key) < 20:
        warnings.append(f"{exchange} API key seems too short ({len(api_key)} chars)")

    if not api_secret:
        warnings.append(f"{exchange} API secret is not set")
    elif len(api_secret) < 20:
        warnings.append(f"{exchange} API secret seems too short ({len(api_secret)} chars)")

    return warnings
