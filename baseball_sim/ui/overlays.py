"""Ephemeral overlay events for independent on-field announcements.

These are separate from the normal play-by-play (last_play) so that
substitution events (pinch run, pinch hit, defensive sub, pitching change)
can always be shown as independent overlays on the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OverlayEvent:
    type: str
    text: str

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


class OverlayEventCenter:
    def __init__(self) -> None:
        self._events: List[OverlayEvent] = []

    def publish(self, event_type: str, text: str) -> OverlayEvent:
        event = OverlayEvent(type=str(event_type or ""), text=str(text or ""))
        self._events.append(event)
        return event

    def consume(self) -> List[dict]:
        if not self._events:
            return []
        events = [e.to_dict() for e in self._events]
        self._events.clear()
        return events

