"""Formatting helpers shared by the browser session components."""

from __future__ import annotations

from typing import Optional

from baseball_sim.gameplay.game import GameState


def inning_ordinal(inning: int) -> str:
    """Convert an inning number to its ordinal representation."""

    if inning == 1:
        return "1st"
    if inning == 2:
        return "2nd"
    if inning == 3:
        return "3rd"
    return f"{inning}th"


def half_inning_banner(game_state: Optional[GameState], home_team, away_team) -> str:
    """Return the banner text announcing the start of a half inning."""

    if not game_state:
        return ""

    half = "TOP" if game_state.is_top_inning else "BOTTOM"
    ordinal = inning_ordinal(game_state.inning)
    batting_team = away_team.name if game_state.is_top_inning else home_team.name
    emoji = "🔼" if game_state.is_top_inning else "🔽"

    lines = [
        "=" * 40,
        f"=== {emoji} {half} of the {ordinal} ===",
        f"🏃 {batting_team} batting",
    ]

    if game_state.inning >= 9 and not game_state.is_top_inning:
        home_score = getattr(game_state, "home_score", 0)
        away_score = getattr(game_state, "away_score", 0)
        if home_score > away_score:
            lines.append("⚡ Walk-off opportunity!")

    lines.append("=" * 40)
    return "\n".join(lines)


def format_situation(game_state: Optional[GameState]) -> str:
    """Create a short textual summary of the current game situation."""

    if not game_state:
        return ""

    bases = []
    if game_state.bases[0]:
        bases.append("1st")
    if game_state.bases[1]:
        bases.append("2nd")
    if game_state.bases[2]:
        bases.append("3rd")

    runner_text = "Bases Empty" if not bases else f"Runners on {', '.join(bases)}"
    half = "Top" if game_state.is_top_inning else "Bottom"
    return f"{half} {game_state.inning} — {game_state.outs} Outs — {runner_text}"


def format_matchup(
    batter: Optional[dict[str, object]],
    pitcher: Optional[dict[str, object]],
) -> Optional[str]:
    """Return a string describing the current batter/pitcher matchup."""

    if not batter or not pitcher:
        return None
    return f"{batter['name']} vs {pitcher['name']}"


def count_hits(team) -> int:
    """Compute the hit total for a team based on player stats."""

    if not team:
        return 0

    total = 0
    for player in getattr(team, "lineup", []):
        stats = getattr(player, "stats", {}) or {}
        singles = stats.get("1B", 0)
        doubles = stats.get("2B", 0)
        triples = stats.get("3B", 0)
        homers = stats.get("HR", 0)
        total += singles + doubles + triples + homers
    return int(total)

