"""Utilities for preparing teams and schedules for league simulations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from baseball_sim.config import get_project_paths, setup_project_environment
from baseball_sim.infrastructure.logging_utils import logger as root_logger

setup_project_environment()
PATHS = get_project_paths()
LOGGER = root_logger.getChild("interface.league_setup")


@dataclass
class TeamContext:
    """Per-team state that needs to persist across games in a league."""

    team: object
    original_name: str
    unique_name: str
    pitcher_pool: List[object]
    rotation: List[object]
    rotation_index: int = 0

    def next_starting_pitcher(self) -> Optional[object]:
        pitcher = select_starting_pitcher(self.rotation, self.rotation_index)
        if self.rotation:
            self.rotation_index = (self.rotation_index + 1) % len(self.rotation)
        return pitcher


def get_pitcher_rotation(pitchers, preferred_rotation=None):
    """Build a starting rotation list from the available pitchers."""

    if not pitchers:
        return []

    if preferred_rotation:
        rotation_list = []
        seen = set()
        for pitcher in preferred_rotation:
            if pitcher and pitcher in pitchers and id(pitcher) not in seen:
                rotation_list.append(pitcher)
                seen.add(id(pitcher))
        if rotation_list:
            return rotation_list

    starters = [
        pitcher
        for pitcher in pitchers
        if getattr(pitcher, "pitcher_type", "").upper() == "SP"
    ]

    return list(starters)


def select_starting_pitcher(rotation, index):
    """Return the starting pitcher for a rotation index."""

    if not rotation:
        return None

    safe_index = index % len(rotation)
    return rotation[safe_index]


def _assign_unique_team_names(contexts: Sequence[TeamContext]) -> None:
    """Assign a unique name to each team when duplicates exist."""

    totals: Dict[str, int] = defaultdict(int)
    for ctx in contexts:
        totals[ctx.original_name] += 1

    counters: Dict[str, int] = defaultdict(int)
    for ctx in contexts:
        base = ctx.original_name
        if totals[base] > 1:
            counters[base] += 1
            unique = f"{base}-{counters[base]}"
        else:
            unique = base
        ctx.unique_name = unique
        if hasattr(ctx.team, "name"):
            ctx.team.name = unique


def build_team_contexts(team_datas: Sequence[Mapping[str, object]]) -> List[TeamContext]:
    """Create team contexts used to coordinate league simulations."""

    if not team_datas:
        raise ValueError("League simulation requires at least two teams.")

    from baseball_sim.data.loader import DataLoader

    player_data_path = PATHS.get_players_data_path()
    player_data = DataLoader.load_json_data(player_data_path)

    contexts: List[TeamContext] = []
    for data in team_datas:
        if not isinstance(data, Mapping):
            raise ValueError("Each team configuration must be a mapping object.")
        team_config = dict(data)
        team, warnings = DataLoader.create_team(team_config, player_data=player_data)
        for warning in warnings:
            LOGGER.warning("Team setup warning: %s", warning)
        try:
            initial_lineup = list(getattr(team, "lineup", []) or [])
            initial_positions = [getattr(p, "current_position", None) for p in initial_lineup]
            setattr(team, "_initial_lineup", initial_lineup)
            setattr(team, "_initial_lineup_positions", initial_positions)
            setattr(team, "_initial_bench", list(getattr(team, "bench", []) or []))
        except Exception:
            pass
        context = TeamContext(
            team=team,
            original_name=team_config.get("name", getattr(team, "name", "Team")),
            unique_name=getattr(team, "name", "Team"),
            pitcher_pool=list(team.pitchers),
            rotation=get_pitcher_rotation(list(team.pitchers), getattr(team, "pitcher_rotation", None)),
        )
        contexts.append(context)

    _assign_unique_team_names(contexts)
    return contexts


def initialize_league_results(
    contexts: Sequence[TeamContext],
    *,
    role_assignment: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    """Initialise the shared results structure for a league run."""

    results: Dict[str, object] = {
        "games": [],
        "team_stats": {},
        "teams": {},
        "meta": {},
        "team_aliases": {},
    }

    for index, ctx in enumerate(contexts):
        team = ctx.team
        team_name = getattr(team, "name", f"Team {index + 1}")
        original_name = ctx.original_name

        results["teams"][team_name] = team
        if original_name not in results["teams"]:
            results["teams"][original_name] = team

        if role_assignment:
            for role, assigned_index in role_assignment.items():
                if assigned_index != index:
                    continue
                results["teams"][role] = team
                suffix = "Home" if role == "home" else "Away"
                alias_name = f"{original_name} ({suffix})"
                results["teams"][alias_name] = team
                results["team_aliases"][alias_name] = team_name

    return results


def generate_round_robin_pairs(num_teams: int) -> List[List[Tuple[int, int]]]:
    """Generate pairings for a round robin schedule with an even number of teams."""

    if num_teams % 2 != 0:
        raise ValueError("Team count must be even for league scheduling.")

    if num_teams < 2:
        return []

    indices = list(range(num_teams))
    half = num_teams // 2
    rounds: List[List[Tuple[int, int]]] = []

    for round_index in range(num_teams - 1):
        pairs: List[Tuple[int, int]] = []
        for i in range(half):
            first = indices[i]
            second = indices[-(i + 1)]
            if round_index % 2 == 0:
                home, away = first, second
            else:
                home, away = second, first
            pairs.append((home, away))
        rounds.append(pairs)

        indices = [indices[0]] + [indices[-1]] + indices[1:-1]

    return rounds


__all__ = [
    "TeamContext",
    "build_team_contexts",
    "generate_round_robin_pairs",
    "get_pitcher_rotation",
    "initialize_league_results",
    "select_starting_pitcher",
]
