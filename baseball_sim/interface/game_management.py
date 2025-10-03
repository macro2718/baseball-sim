"""Helpers for preparing teams and tracking results between games."""

from __future__ import annotations

def reset_team_and_players(
    home_team,
    away_team,
    *,
    home_pitcher_pool=None,
    away_pitcher_pool=None,
    home_starting_pitcher=None,
    away_starting_pitcher=None,
):
    """Reset team/player state between games in a series."""

    for team in (home_team, away_team):
        if hasattr(team, "ejected_players"):
            team.ejected_players = []
        try:
            init_lineup = list(getattr(team, "_initial_lineup", []) or [])
            init_positions = list(getattr(team, "_initial_lineup_positions", []) or [])
            init_bench = list(getattr(team, "_initial_bench", []) or [])

            if init_lineup and init_positions and len(init_lineup) == len(init_positions):
                team.lineup = list(init_lineup)
                if hasattr(team, "defensive_positions"):
                    for key in list(team.defensive_positions.keys()):
                        team.defensive_positions[key] = None
                for player, pos in zip(init_lineup, init_positions):
                    if pos is not None:
                        player.current_position = pos
                        if hasattr(team, "defensive_positions"):
                            team.defensive_positions[pos] = player
                team.bench = list(init_bench)
        except Exception:
            pass

    if home_pitcher_pool is not None:
        home_team.pitchers = list(home_pitcher_pool)
    if away_pitcher_pool is not None:
        away_team.pitchers = list(away_pitcher_pool)

    if hasattr(home_team, "pitcher_rotation"):
        home_team.pitcher_rotation = [
            pitcher for pitcher in home_team.pitcher_rotation if pitcher in home_team.pitchers
        ]
    if hasattr(away_team, "pitcher_rotation"):
        away_team.pitcher_rotation = [
            pitcher for pitcher in away_team.pitcher_rotation if pitcher in away_team.pitchers
        ]

    for pitcher in home_team.pitchers:
        ptype = getattr(pitcher, "pitcher_type", "").upper()
        if ptype == "RP":
            current = getattr(pitcher, "current_stamina", pitcher.stamina)
            pitcher.current_stamina = min(pitcher.stamina, current + 10)
        else:
            pitcher.current_stamina = pitcher.stamina

    for pitcher in away_team.pitchers:
        ptype = getattr(pitcher, "pitcher_type", "").upper()
        if ptype == "RP":
            current = getattr(pitcher, "current_stamina", pitcher.stamina)
            pitcher.current_stamina = min(pitcher.stamina, current + 10)
        else:
            pitcher.current_stamina = pitcher.stamina

    home_team.current_batter_index = 0
    away_team.current_batter_index = 0

    home_team.current_pitcher = (
        home_starting_pitcher
        if home_starting_pitcher is not None
        else (home_team.pitchers[0] if home_team.pitchers else None)
    )
    away_team.current_pitcher = (
        away_starting_pitcher
        if away_starting_pitcher is not None
        else (away_team.pitchers[0] if away_team.pitchers else None)
    )


def update_statistics(results, game, game_result):
    """Update aggregated team statistics with the outcome of a game."""

    def ensure_entry(team_obj):
        team_name = getattr(team_obj, "name", "Team")
        stats = results.setdefault("team_stats", {}).setdefault(
            team_name,
            {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "runs_scored": 0,
                "runs_allowed": 0,
                "games": 0,
            },
        )
        return team_name, stats

    home_name, home_stats = ensure_entry(game.home_team)
    away_name, away_stats = ensure_entry(game.away_team)

    home_stats["games"] += 1
    away_stats["games"] += 1

    if game.home_score > game.away_score:
        home_stats["wins"] += 1
        away_stats["losses"] += 1
    elif game.away_score > game.home_score:
        home_stats["losses"] += 1
        away_stats["wins"] += 1
    else:
        home_stats["draws"] += 1
        away_stats["draws"] += 1

    home_stats["runs_scored"] += game.home_score
    home_stats["runs_allowed"] += game.away_score
    away_stats["runs_scored"] += game.away_score
    away_stats["runs_allowed"] += game.home_score

    aliases = results.get("team_aliases", {})
    for alias, target in aliases.items():
        if target == home_name:
            results["team_stats"][alias] = home_stats
        elif target == away_name:
            results["team_stats"][alias] = away_stats


__all__ = ["reset_team_and_players", "update_statistics"]
