"""Utilities for working with pitcher rotation data."""

from __future__ import annotations

from typing import Iterable, List


def parse_rotation_entries(raw_rotation: object) -> List[str]:
    """Extract trimmed pitcher names from rotation entries.

    The rotation data can contain either string names or dictionaries with a
    ``name`` field. Any other values are ignored. Empty names are filtered out
    after stripping surrounding whitespace.
    """

    if not isinstance(raw_rotation, Iterable) or isinstance(raw_rotation, (str, bytes)):
        return []

    names: List[str] = []
    for entry in raw_rotation:
        if isinstance(entry, dict):
            name_value = entry.get("name")
        else:
            name_value = entry
        name = str(name_value or "").strip()
        if name:
            names.append(name)
    return names


__all__ = ["parse_rotation_entries"]
