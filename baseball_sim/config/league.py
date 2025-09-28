"""League-average rates used across the simulator.

This module centralizes league-average percentages (K%, BB%, Hard%, GB%).
Values are expressed as percentages (e.g., 22.8 for 22.8%).

Sources of truth:
- Defaults live in ConfigManager (settings.py) under key "league.averages".
- Users may override via baseball_sim/config/config.json with the same keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from .settings import config


@dataclass(frozen=True)
class LeagueAverages:
    k_pct: float
    bb_pct: float
    hard_pct: float
    gb_pct: float

    @staticmethod
    def load() -> "LeagueAverages":
        """Load league-average values from configuration (with defaults)."""
        data = config.get("league.averages", {}) or {}
        # Fallbacks mirror historical MLB(2021-2023) blended baselines used in the app
        return LeagueAverages(
            k_pct=float(data.get("k_pct", 22.8)),
            bb_pct=float(data.get("bb_pct", 8.5)),
            hard_pct=float(data.get("hard_pct", 38.6)),
            gb_pct=float(data.get("gb_pct", 44.6)),
        )

    def as_dict(self) -> dict:
        return {
            "k_pct": self.k_pct,
            "bb_pct": self.bb_pct,
            "hard_pct": self.hard_pct,
            "gb_pct": self.gb_pct,
        }

