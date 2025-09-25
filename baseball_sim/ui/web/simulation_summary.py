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
    num_games: int,
) -> Dict[str, Any]:
    """Convert simulator output to the structure consumed by the web UI."""

    team_objects = _extract_team_objects(results)
    team_stats = results.get("team_stats") or {}

    teams_summary = [
        _build_team_entry("away", away_team, team_objects, team_stats),
        _build_team_entry("home", home_team, team_objects, team_stats),
    ]

    games, recent_games = _build_game_summaries(results.get("games") or [])

    return {
        "total_games": int(num_games),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "teams": teams_summary,
        "games": games,
        "recent_games": recent_games,
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
    team_key: str,
    fallback_team: Optional[object],
    team_objects: Mapping[str, object],
    team_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    team_name = getattr(fallback_team, "name", None) or team_key.title()
    team_obj = team_objects.get(team_name, fallback_team)

    return {
        "key": team_key,
        "name": team_name,
        "record": _build_team_record(team_name, team_stats),
        "batting": _compute_team_batting(team_obj),
        "pitching": _compute_team_pitching(team_obj),
        "batters": _build_batter_stats(team_obj),
        "pitchers": _build_pitcher_stats(team_obj),
    }


def _build_team_record(team_name: str, team_stats: Mapping[str, Any]) -> Dict[str, Any]:
    stats = team_stats.get(team_name, {})
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
    }


def _iter_team_hitters(team_obj: Optional[object]) -> Iterable[object]:
    if not team_obj:
        return []

    players: List[object] = []
    seen_ids = set()
    for collection in (
        getattr(team_obj, "lineup", []) or [],
        getattr(team_obj, "bench", []) or [],
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

    for pitcher in getattr(team_obj, "pitchers", []) or []:
        stats = getattr(pitcher, "pitching_stats", {}) or {}
        totals["ip"] += float(stats.get("IP", 0) or 0.0)
        totals["hits_allowed"] += int(stats.get("H", 0) or 0)
        totals["runs_allowed"] += int(stats.get("R", 0) or 0)
        totals["earned_runs"] += int(stats.get("ER", 0) or 0)
        totals["walks"] += int(stats.get("BB", 0) or 0)
        totals["strikeouts"] += int(stats.get("SO", stats.get("K", 0)) or 0)
        totals["home_runs"] += int(stats.get("HR", 0) or 0)

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
    for pitcher in getattr(team_obj, "pitchers", []) or []:
        stats = getattr(pitcher, "pitching_stats", {}) or {}
        ip = float(stats.get("IP", 0) or 0.0)
        hits = int(stats.get("H", 0) or 0)
        runs = int(stats.get("R", 0) or 0)
        earned = int(stats.get("ER", 0) or 0)
        walks = int(stats.get("BB", 0) or 0)
        strikeouts = int(stats.get("SO", stats.get("K", 0)) or 0)
        homers = int(stats.get("HR", 0) or 0)

        era, whip, k_per_9, bb_per_9 = _safe_pitching_metrics(pitcher)
        hr_per_9 = StatsCalculator.calculate_hr_per_9(homers, ip)

        pitchers.append(
            {
                "name": getattr(pitcher, "name", ""),
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
            }
        )

    recent_games = all_games[-5:] if all_games else []
    return all_games, recent_games

