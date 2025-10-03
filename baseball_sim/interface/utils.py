"""Utility helpers for interface-related modules."""

from __future__ import annotations

from typing import Iterable, Mapping


def iter_unique_teams(results: Mapping[str, object]) -> Iterable[object]:
    """Yield unique team objects present in the results mapping."""

    teams = results.get("teams") or {}
    seen_ids: set[int] = set()
    for team in teams.values():
        if team is None:
            continue
        identifier = id(team)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        yield team


__all__ = ["iter_unique_teams"]
