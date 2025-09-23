"""Helpers for managing the rolling in-game log displayed in the UI."""

from __future__ import annotations

from typing import Dict, Iterable, List


class SessionLog:
    """Maintain the capped list of play-by-play log entries."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: List[Dict[str, str]] = []

    def append(self, message: str, variant: str = "info") -> None:
        if not message:
            return
        self._entries.append({"text": message, "variant": variant})
        if len(self._entries) > self._max_entries:
            del self._entries[:-self._max_entries]

    def extend_banner(self, banner: str) -> None:
        """Append a multi-line banner, inferring variants for each line."""

        for line in banner.splitlines():
            if not line.strip():
                continue
            variant = "highlight" if line.strip().startswith("=") else "info"
            self.append(line, variant=variant)

    def clear(self) -> None:
        self._entries.clear()

    def as_list(self) -> List[Dict[str, str]]:
        return list(self._entries)

    def update(self, entries: Iterable[Dict[str, str]]) -> None:
        """Replace the log contents, keeping only the newest entries."""

        self._entries = []
        for entry in entries:
            text = entry.get("text", "") if isinstance(entry, dict) else ""
            variant = entry.get("variant", "info") if isinstance(entry, dict) else "info"
            self.append(text, variant)

