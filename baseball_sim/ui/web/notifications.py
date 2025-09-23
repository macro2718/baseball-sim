"""Notification helpers for the web UI session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Notification:
    """Represents a single message destined for the frontend."""

    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


class NotificationCenter:
    """Store and release one-off notifications for the UI."""

    def __init__(self) -> None:
        self._current: Optional[Notification] = None

    def publish(self, level: str, message: str) -> Notification:
        """Create and store a notification for later retrieval."""

        notification = Notification(level=level, message=message)
        self._current = notification
        return notification

    def set(self, notification: Optional[Notification]) -> None:
        """Directly replace the stored notification."""

        self._current = notification

    def clear(self) -> None:
        """Discard any stored notification."""

        self._current = None

    def peek(self) -> Optional[Notification]:
        """Return the stored notification without clearing it."""

        return self._current

    def consume(self) -> Optional[Notification]:
        """Return and clear the stored notification."""

        notification = self._current
        self._current = None
        return notification
