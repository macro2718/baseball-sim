"""Helpers for converting raw simulation output into UI friendly data."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from baseball_sim.gameplay.statistics import StatsCalculator


def summarize_simulation_results(
    results: Mapping[str, Any],
    *,
    home_team: Optional[object],
    away_team: Optional[object],
    num_games: Optional[int],
    league: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert simulator output to the structure consumed by the web UI."""

    team_objects = _extract_team_objects(results)
    team_stats = results.get("team_stats") or {}
    alias_map = results.get("team_aliases") or {}

    unique_teams = list(_iter_unique_teams(results))
    league_mode = bool(league) or len(unique_teams) > 2

    role_map: Dict[str, str] = {}
    for role in ("away", "home"):
        candidate = team_objects.get(role)
        if candidate is not None:
            name = getattr(candidate, "name", None)
            if name:
                role_map[role] = name

    team_ids = []
    if league and isinstance(league, Mapping):
        raw_ids = league.get("teams")
        if isinstance(raw_ids, list):
            team_ids = [str(team_id) for team_id in raw_ids]

    team_entries: List[Dict[str, Any]] = []
    for index, team_obj in enumerate(unique_teams):
        summary = _build_team_entry(
            index,
            team_obj,
            team_objects,
            team_stats,
            alias_map,
            role_map,
            team_ids,
            league_mode,
        )
        team_entries.append(summary)

    if league_mode:
        team_entries.sort(
            key=lambda entry: (
                entry["record"].get("wins", 0),
                entry["record"].get("runDiff", 0),
                entry["record"].get("runsScored", 0),
            ),
            reverse=True,
        )
    else:
        ordered: List[Dict[str, Any]] = []
        for key in ("away", "home"):
            entry = next((team for team in team_entries if key in team.get("roles", []) or team.get("key") == key), None)
            if entry and entry not in ordered:
                ordered.append(entry)
        for entry in team_entries:
            if entry not in ordered:
                ordered.append(entry)
        team_entries = ordered

    for position, entry in enumerate(team_entries, start=1):
        entry["rank"] = position if league_mode else None

    games, recent_games = _build_game_summaries(results.get("games") or [])

    meta = results.get("meta") or {}
    fallback_games = num_games if isinstance(num_games, int) and num_games > 0 else 0
    total_games = int(meta.get("completed_games") or meta.get("scheduled_games") or fallback_games or len(games))
    league_meta = {
        "total_teams": int(meta.get("total_teams") or len(unique_teams)),
        "games_per_card": int(meta.get("games_per_card", 0) or 0) or None,
        "cards_per_opponent": int(meta.get("cards_per_opponent", 0) or 0) or None,
        "completed_days": int(meta.get("completed_days", 0) or 0) or None,
        "total_days": int(meta.get("total_days", 0) or 0) or None,
        "completed_games": int(meta.get("completed_games", 0) or 0) or None,
        "scheduled_games": int(meta.get("scheduled_games", 0) or 0) or None,
        "teams": team_ids,
    }
    if league:
        if league.get("games_per_card"):
            league_meta["games_per_card"] = league.get("games_per_card")
        if league.get("cards_per_opponent"):
            league_meta["cards_per_opponent"] = league.get("cards_per_opponent")
        if league.get("teams"):
            league_meta["teams"] = list(league.get("teams"))

    return {
        "total_games": total_games,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "teams": team_entries,
        "games": games,
        "recent_games": recent_games,
        "mode": "league" if league_mode else "series",
        "league": league_meta,
        "aliases": alias_map,
        "roles": role_map,
    }


def _extract_team_objects(results: Mapping[str, Any]) -> Mapping[str, object]:
    raw = results.get("teams") or {}
    named = {}
    for name, team in raw.items():
        named[name] = team
        team_name = getattr(team, "name", None)
        if team_name and team_name not in named:
            named[team_name] = team
    return named


def _build_team_entry(
    index: int,
    team_obj: Optional[object],
    team_objects: Mapping[str, object],
    team_stats: Mapping[str, Any],
    alias_map: Mapping[str, str],
    role_map: Mapping[str, str],
    team_ids: List[str],
    league_mode: bool,
) -> Dict[str, Any]:
    team_name = getattr(team_obj, "name", None) or f"Team {index + 1}"
    resolved_roles = [role for role, name in role_map.items() if name == team_name]
    resolved_aliases = [alias for alias, target in alias_map.items() if target == team_name]
    if not team_obj and team_name in team_objects:
        team_obj = team_objects.get(team_name)

    key = resolved_roles[0] if resolved_roles else f"team-{index + 1}"
    record = _build_team_record(team_name, team_stats, role=resolved_roles[0] if resolved_roles else None)
    batting = _compute_team_batting(team_obj)
    pitching = _compute_team_pitching(team_obj)
    batters = _build_batter_stats(team_obj)
    pitchers = _build_pitcher_stats(team_obj)
    team_id = team_ids[index] if index < len(team_ids) else None

    display_name = team_name
    if resolved_roles and not league_mode:
        role_label = 'Away' if resolved_roles[0] == 'away' else 'Home'
        display_name = f"{team_name} ({role_label})"

    return {
        "key": key,
        "id": team_id,
        "name": display_name,
        "roles": resolved_roles,
        "aliases": resolved_aliases,
        "record": record,
        "batting": batting,
        "pitching": pitching,
        "batters": batters,
        "pitchers": pitchers,
        "rank": index + 1,
    }


def _build_team_record(team_name: str, team_stats: Mapping[str, Any], *, role: Optional[str] = None) -> Dict[str, Any]:
    stats = team_stats.get(team_name, {})
    if (not stats) and role in {"home", "away"}:
        suffix = "Home" if role == "home" else "Away"
        role_key = f"{team_name} ({suffix})"
        stats = team_stats.get(role_key, {})
    wins = int(stats.get("wins", 0))
    losses = int(stats.get("losses", 0))
    draws = int(stats.get("draws", 0))
    runs_scored = int(stats.get("runs_scored", 0))
    runs_allowed = int(stats.get("runs_allowed", 0))
    total = max(wins + losses + draws, 0)
    win_pct = wins / total if total else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "runs_scored": runs_scored,
        "runs_allowed": runs_allowed,
        "run_diff": runs_scored - runs_allowed,
        "win_pct": win_pct,
        "games": int(stats.get("games", total)),
    }


def _iter_team_hitters(team_obj: Optional[object]) -> Iterable[object]:
    if not team_obj:
        return []

    players: List[object] = []
    seen_ids = set()
    # Include current lineup, remaining bench, and ejected players
    # so that batters who left via PH/PR still appear in stats.
    for collection in (
        getattr(team_obj, "lineup", []) or [],
        getattr(team_obj, "bench", []) or [],
        getattr(team_obj, "ejected_players", []) or [],
    ):
        for player in collection:
            if player is None:
                continue
            identifier = id(player)
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            players.append(player)
    return players


def _compute_team_batting(team_obj: Optional[object]) -> Dict[str, Any]:
    totals = {
        "pa": 0,
        "ab": 0,
        "singles": 0,
        "doubles": 0,
        "triples": 0,
        "home_runs": 0,
        "walks": 0,
        "strikeouts": 0,
        "hits": 0,
    }

    for player in _iter_team_hitters(team_obj):
        stats = getattr(player, "stats", {}) or {}
        singles = int(stats.get("1B", 0) or 0)
        doubles = int(stats.get("2B", 0) or 0)
        triples = int(stats.get("3B", 0) or 0)
        homers = int(stats.get("HR", 0) or 0)
        walks = int(stats.get("BB", 0) or 0)
        strikeouts = int(stats.get("SO", stats.get("K", 0)) or 0)
        plate_appearances = int(stats.get("PA", 0) or 0)
        at_bats = int(stats.get("AB", 0) or 0)

        totals["pa"] += plate_appearances
        totals["ab"] += at_bats
        totals["singles"] += singles
        totals["doubles"] += doubles
        totals["triples"] += triples
        totals["home_runs"] += homers
        totals["walks"] += walks
        totals["strikeouts"] += strikeouts
        totals["hits"] += singles + doubles + triples + homers

    avg = totals["hits"] / totals["ab"] if totals["ab"] > 0 else 0.0
    obp = StatsCalculator.calculate_obp(totals["hits"], totals["walks"], totals["ab"])
    slg = StatsCalculator.calculate_slg(
        totals["singles"],
        totals["doubles"],
        totals["triples"],
        totals["home_runs"],
        totals["ab"],
    )
    ops = StatsCalculator.calculate_ops(obp, slg)

    totals.update({"avg": avg, "obp": obp, "slg": slg, "ops": ops})
    return totals


def _compute_team_pitching(team_obj: Optional[object]) -> Dict[str, Any]:
    totals = {
        "ip": 0.0,
        "hits_allowed": 0,
        "runs_allowed": 0,
        "earned_runs": 0,
        "walks": 0,
        "strikeouts": 0,
        "home_runs": 0,
    }

    # Build a unique set of pitchers who have or could have pitched
    seen: set[int] = set()

    def add_pitcher(p) -> None:
        if p is None:
            return
        pid = id(p)
        if pid in seen:
            return
        seen.add(pid)
        stats = getattr(p, "pitching_stats", {}) or {}
        totals["ip"] += float(stats.get("IP", 0) or 0.0)
        totals["hits_allowed"] += int(stats.get("H", 0) or 0)
        totals["runs_allowed"] += int(stats.get("R", 0) or 0)
        totals["earned_runs"] += int(stats.get("ER", 0) or 0)
        totals["walks"] += int(stats.get("BB", 0) or 0)
        totals["strikeouts"] += int(stats.get("SO", stats.get("K", 0)) or 0)
        totals["home_runs"] += int(stats.get("HR", 0) or 0)

    add_pitcher(getattr(team_obj, "current_pitcher", None))
    for p in getattr(team_obj, "pitchers", []) or []:
        add_pitcher(p)
    for p in getattr(team_obj, "ejected_players", []) or []:
        if hasattr(p, "pitcher_type") or hasattr(p, "pitching_stats"):
            add_pitcher(p)

    era = StatsCalculator.calculate_era(totals["earned_runs"], totals["ip"])
    whip = StatsCalculator.calculate_whip(
        totals["hits_allowed"], totals["walks"], totals["ip"]
    )
    k_per_9 = StatsCalculator.calculate_k_per_9(totals["strikeouts"], totals["ip"])
    bb_per_9 = StatsCalculator.calculate_bb_per_9(totals["walks"], totals["ip"])
    hr_per_9 = StatsCalculator.calculate_hr_per_9(totals["home_runs"], totals["ip"])

    totals.update(
        {
            "era": era,
            "whip": whip,
            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
            "hr_per_9": hr_per_9,
        }
    )
    return totals


def _build_batter_stats(team_obj: Optional[object]) -> List[Dict[str, Any]]:
    batters: List[Dict[str, Any]] = []
    for player in _iter_team_hitters(team_obj):
        stats = getattr(player, "stats", {}) or {}
        pa = int(stats.get("PA", 0) or 0)
        if pa == 0:
            continue
        ab = int(stats.get("AB", 0) or 0)
        singles = int(stats.get("1B", 0) or 0)
        doubles = int(stats.get("2B", 0) or 0)
        triples = int(stats.get("3B", 0) or 0)
        homers = int(stats.get("HR", 0) or 0)
        walks = int(stats.get("BB", 0) or 0)
        strikeouts = int(stats.get("SO", stats.get("K", 0)) or 0)
        runs = int(stats.get("R", 0) or 0)
        rbi = int(stats.get("RBI", 0) or 0)
        hits = singles + doubles + triples + homers

        avg, obp, slg, ops = _safe_hitting_metrics(player)
        k_pct = (strikeouts / pa * 100) if pa > 0 else 0.0
        bb_pct = (walks / pa * 100) if pa > 0 else 0.0

        batters.append(
            {
                "name": getattr(player, "name", ""),
                "pa": pa,
                "ab": ab,
                "singles": singles,
                "doubles": doubles,
                "triples": triples,
                "home_runs": homers,
                "runs": runs,
                "rbi": rbi,
                "walks": walks,
                "strikeouts": strikeouts,
                "hits": hits,
                "avg": avg,
                "obp": obp,
                "slg": slg,
                "ops": ops,
                "k_pct": k_pct,
                "bb_pct": bb_pct,
            }
        )

    return batters


def _safe_hitting_metrics(player: object) -> tuple:
    try:
        avg = float(player.get_avg()) if hasattr(player, "get_avg") else 0.0
        obp = float(player.get_obp()) if hasattr(player, "get_obp") else 0.0
        slg = float(player.get_slg()) if hasattr(player, "get_slg") else 0.0
        ops = float(player.get_ops()) if hasattr(player, "get_ops") else 0.0
    except Exception:  # pragma: no cover - defensive
        avg = obp = slg = ops = 0.0
    return avg, obp, slg, ops


def _build_pitcher_stats(team_obj: Optional[object]) -> List[Dict[str, Any]]:
    pitchers: List[Dict[str, Any]] = []

    def append_pitcher(pitcher) -> None:
        stats = getattr(pitcher, "pitching_stats", {}) or {}
        ip = float(stats.get("IP", 0) or 0.0)
        hits = int(stats.get("H", 0) or 0)
        runs = int(stats.get("R", 0) or 0)
        earned = int(stats.get("ER", 0) or 0)
        walks = int(stats.get("BB", 0) or 0)
        strikeouts = int(stats.get("SO", stats.get("K", 0)) or 0)
        homers = int(stats.get("HR", 0) or 0)
        games = int(stats.get("G", 0) or 0)

        era, whip, k_per_9, bb_per_9 = _safe_pitching_metrics(pitcher)
        hr_per_9 = StatsCalculator.calculate_hr_per_9(homers, ip)

        pitchers.append(
            {
                "name": getattr(pitcher, "name", ""),
                "appearances": games,
                "ip": ip,
                "hits": hits,
                "runs": runs,
                "earned_runs": earned,
                "walks": walks,
                "strikeouts": strikeouts,
                "home_runs": homers,
                "era": era,
                "whip": whip,
                "k_per_9": k_per_9,
                "bb_per_9": bb_per_9,
                "hr_per_9": hr_per_9,
            }
        )

    seen: set[int] = set()

    def add_unique(p) -> None:
        if p is None:
            return
        pid = id(p)
        if pid in seen:
            return
        seen.add(pid)
        append_pitcher(p)

    add_unique(getattr(team_obj, "current_pitcher", None))
    for p in getattr(team_obj, "pitchers", []) or []:
        add_unique(p)
    for p in getattr(team_obj, "ejected_players", []) or []:
        if hasattr(p, "pitcher_type") or hasattr(p, "pitching_stats"):
            add_unique(p)

    return pitchers


def _safe_pitching_metrics(pitcher: object) -> tuple:
    try:
        era = float(pitcher.get_era()) if hasattr(pitcher, "get_era") else 0.0
        whip = float(pitcher.get_whip()) if hasattr(pitcher, "get_whip") else 0.0
        k_per_9 = (
            float(pitcher.get_k_per_9()) if hasattr(pitcher, "get_k_per_9") else 0.0
        )
        bb_per_9 = (
            float(pitcher.get_bb_per_9()) if hasattr(pitcher, "get_bb_per_9") else 0.0
        )
    except Exception:  # pragma: no cover - defensive
        era = whip = k_per_9 = bb_per_9 = 0.0
    return era, whip, k_per_9, bb_per_9


def _build_game_summaries(
    games: Iterable[Mapping[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_games: List[Dict[str, Any]] = []
    for index, game in enumerate(games, start=1):
        try:
            home_score = int(game.get("home_score", 0))
            away_score = int(game.get("away_score", 0))
        except (TypeError, ValueError):  # pragma: no cover - fallback
            home_score, away_score = 0, 0

        if home_score > away_score:
            winner = "home"
        elif home_score < away_score:
            winner = "away"
        else:
            winner = "draw"

        all_games.append(
            {
                "index": index,
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "home_score": home_score,
                "away_score": away_score,
                "innings": int(game.get("innings", 0) or 0),
                "winner": winner,
                "day": int(game.get("day", 0) or 0) or None,
                "card": int(game.get("card", 0) or 0) or None,
                "round": int(game.get("round", 0) or 0) or None,
                "card_game": int(game.get("card_game", 0) or 0) or None,
            }
        )

    recent_games = all_games[-5:] if all_games else []
    return all_games, recent_games
