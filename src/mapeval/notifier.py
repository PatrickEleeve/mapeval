"""Notification system for trading events.

Supports Telegram, Webhook, and composite (multi-channel) notification delivery.
Integrates with the event bus to automatically notify on key trading events.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _requests
except ImportError:
    _requests = None


class Notifier(ABC):
    """Abstract base for notification delivery."""

    @abstractmethod
    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a notification. Returns True if delivered successfully."""
        ...


class LogNotifier(Notifier):
    """Fallback notifier that logs to Python logging."""

    def send_alert(self, level: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, "[ALERT] %s: %s", title, message)
        return True


class TelegramNotifier(Notifier):
    """Send notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not chat_id:
            raise ValueError("bot_token and chat_id are required")
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._api_base = f"https://api.telegram.org/bot{bot_token}"

    def send_alert(self, level: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        if _requests is None:
            logger.error("requests library required for TelegramNotifier")
            return False

        level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "error": "❌"}.get(level, "📌")
        text = f"{level_emoji} *{title}*\n{message}"
        if data:
            for key, value in data.items():
                text += f"\n• {key}: `{value}`"

        try:
            resp = _requests.post(
                f"{self._api_base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Telegram notification failed: %s", exc)
            return False


class WebhookNotifier(Notifier):
    """Send notifications to a generic webhook URL."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}

    def send_alert(self, level: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        if _requests is None:
            logger.error("requests library required for WebhookNotifier")
            return False

        payload = {
            "level": level,
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        try:
            resp = _requests.post(self._url, json=payload, headers=self._headers, timeout=10)
            return 200 <= resp.status_code < 300
        except Exception as exc:
            logger.error("Webhook notification failed: %s", exc)
            return False


class CompositeNotifier(Notifier):
    """Fan-out notifications to multiple channels."""

    def __init__(self, notifiers: Optional[List[Notifier]] = None) -> None:
        self._notifiers = notifiers or []

    def add(self, notifier: Notifier) -> None:
        self._notifiers.append(notifier)

    def send_alert(self, level: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        results = []
        for notifier in self._notifiers:
            try:
                results.append(notifier.send_alert(level, title, message, data))
            except Exception as exc:
                logger.error("Notifier %s failed: %s", type(notifier).__name__, exc)
                results.append(False)
        return any(results) if results else False


def create_notifier(config: Dict[str, Any]) -> Notifier:
    """Factory function to create the appropriate notifier from configuration.

    Config example::

        {
            "telegram": {"bot_token": "...", "chat_id": "..."},
            "webhook": {"url": "https://..."},
        }

    Always includes a LogNotifier as fallback.
    """
    composite = CompositeNotifier()
    composite.add(LogNotifier())

    telegram_config = config.get("telegram", {})
    if telegram_config.get("bot_token") and telegram_config.get("chat_id"):
        try:
            composite.add(TelegramNotifier(
                bot_token=telegram_config["bot_token"],
                chat_id=telegram_config["chat_id"],
            ))
        except Exception as exc:
            logger.warning("Failed to initialize Telegram notifier: %s", exc)

    webhook_config = config.get("webhook", {})
    if webhook_config.get("url"):
        try:
            composite.add(WebhookNotifier(
                url=webhook_config["url"],
                headers=webhook_config.get("headers"),
            ))
        except Exception as exc:
            logger.warning("Failed to initialize Webhook notifier: %s", exc)

    return composite
