"""Serialize the Python game session into dictionaries for the frontend."""

from __future__ import annotations

import math

from typing import Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from baseball_sim.gameplay.statistics import StatsCalculator

from .formatting import count_hits, format_matchup, format_situation
from .notifications import Notification

if TYPE_CHECKING:  # pragma: no cover - typing only
    from baseball_sim.gameplay.game import GameState


class SessionProtocol(Protocol):
    """Minimal protocol describing the attributes used by the builder."""

    home_team: Optional[object]
    away_team: Optional[object]
    game_state: Optional["GameState"]


class SessionStateBuilder:
    """Build dictionaries describing the game session for the frontend."""

    def __init__(self, session: "SessionProtocol") -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_state(
        self,
        *,
        log_entries: List[Dict[str, str]],
        notification: Optional[Notification],
        overlays: List[Dict[str, str]],
        action_block_reason: Optional[str],
    ) -> Tuple[Dict[str, object], Optional[str]]:
        title = self._build_title_state()
        teams = self._build_teams_state()
        team_library = self._build_team_library_state()
        game, updated_reason = self._build_game_state(teams, action_block_reason, overlays)
        simulation = self._build_simulation_state()

        payload: Dict[str, object] = {
            "title": title,
            "teams": teams,
            "team_library": team_library,
            "game": game,
            "log": list(log_entries),
            "notification": notification.to_dict() if notification else None,
            "simulation": simulation,
        }
        return payload, updated_reason

    # ------------------------------------------------------------------
    # Title helpers
    # ------------------------------------------------------------------
    def _build_title_state(self) -> Dict[str, object]:
        # During an active game, avoid re-validating lineups on every state build
        # to reduce repeated CPU work. Lineups were validated at game start and
        # any substitutions are handled with their own validation flows.
        if getattr(self._session, "game_state", None):
            return {
                "home": {
                    "name": getattr(getattr(self._session, "home_team", None), "name", "-"),
                    "valid": True,
                    "message": "In-game",
                    "errors": [],
                },
                "away": {
                    "name": getattr(getattr(self._session, "away_team", None), "name", "-"),
                    "valid": True,
                    "message": "In-game",
                    "errors": [],
                },
                "ready": True,
                "hint": "Game in progress.",
            }

        home_status = self._team_status(self._session.home_team)
        away_status = self._team_status(self._session.away_team)
        ready = bool(home_status["valid"] and away_status["valid"])

        if not self._session.home_team or not self._session.away_team:
            hint = "Teams could not be loaded. Check data files."
        elif ready:
            hint = "Lineups ready. Press Start Game to begin."
        else:
            issues = []
            if not home_status["valid"]:
                issues.append(
                    f"{home_status['name']}: {len(home_status['errors'])} issue(s)"
                )
            if not away_status["valid"]:
                issues.append(
                    f"{away_status['name']}: {len(away_status['errors'])} issue(s)"
                )
            hint = "Lineup validation failed: " + ", ".join(issues)

        return {
            "home": home_status,
            "away": away_status,
            "ready": ready,
            "hint": hint,
        }

    def _team_status(self, team) -> Dict[str, object]:
        if not team:
            return {
                "name": "-",
                "valid": False,
                "message": "Team not loaded",
                "errors": [],
            }

        errors = team.validate_lineup()
        valid = len(errors) == 0
        message = "✓ Ready" if valid else f"⚠ {len(errors)} issue(s)"
        return {
            "name": team.name,
            "valid": valid,
            "message": message,
            "errors": errors,
        }

    def _build_team_library_state(self) -> Dict[str, object]:
        if hasattr(self._session, "get_team_library_state"):
            try:
                state = self._session.get_team_library_state()
            except Exception:  # pragma: no cover - defensive fallback
                state = None
            if isinstance(state, dict):
                return state
        return {
            "teams": [],
            "selection": {"home": None, "away": None},
            "ready": False,
            "hint": "チームデータを読み込めませんでした。",
            "active": {"home": None, "away": None},
        }

    def _build_simulation_state(self) -> Dict[str, object]:
        if hasattr(self._session, "get_simulation_state"):
            try:
                state = self._session.get_simulation_state()
            except Exception:  # pragma: no cover - defensive fallback
                state = None
            if isinstance(state, dict):
                return state
        return {
            "enabled": False,
            "running": False,
            "default_games": 20,
            "limits": {"min_games": 1, "max_games": 200},
            "last_run": None,
            "log": [],
        }

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------
    def _build_game_state(
        self,
        teams: Dict[str, Optional[Dict[str, object]]],
        action_block_reason: Optional[str],
        overlays: Optional[List[Dict[str, str]]],
    ) -> Tuple[Dict[str, object], Optional[str]]:
        control_state: Dict[str, object] = {}
        if hasattr(self._session, "get_control_state"):
            try:
                control_state = self._session.get_control_state()
            except Exception:
                control_state = {}

        game_state = self._session.game_state
        if not game_state:
            return (
                {
                    "active": False,
                    "actions": {"swing": False, "bunt": False, "steal": False, "progress": False},
                    "action_block_reason": action_block_reason,
                    "game_over": False,
                    "defensive_errors": [],
                    "score": {"home": 0, "away": 0},
                    "hits": {"home": 0, "away": 0},
                    "errors": {"home": 0, "away": 0},
                    "inning_scores": {"home": [], "away": []},
                    "situation": "Waiting for a new game.",
                    "max_innings": 9,
                    "control": control_state,
                    "overlays": [],
                },
                action_block_reason,
            )

        batting_team = game_state.batting_team
        offense = None
        defense = None
        if batting_team is self._session.home_team:
            offense, defense = "home", "away"
        elif batting_team is self._session.away_team:
            offense, defense = "away", "home"

        current_batter = None
        if batting_team and batting_team.lineup:
            batter = batting_team.current_batter
            current_batter = {
                "name": batter.name,
                "order": batting_team.current_batter_index + 1,
                "position": self._display_position(batter),
                "pitcher_type": getattr(batter, "pitcher_type", None),
            }

        current_pitcher = None
        if game_state.fielding_team and game_state.fielding_team.current_pitcher:
            pitcher = game_state.fielding_team.current_pitcher
            throws = getattr(pitcher, "throws", None)
            throws_display = str(throws).upper() if throws else "-"
            pitcher_type = getattr(pitcher, "pitcher_type", None)
            pitcher_type_display = (
                str(pitcher_type).upper() if pitcher_type else (self._display_position(pitcher) or "P")
            )
            current_pitcher = {
                "name": pitcher.name,
                "stamina": round(getattr(pitcher, "current_stamina", 0), 1),
                "pitcher_type": pitcher_type_display,
                "throws": throws_display,
                "k_pct": self._format_percentage(getattr(pitcher, "k_pct", None)),
                "bb_pct": self._format_percentage(getattr(pitcher, "bb_pct", None)),
                "hard_pct": self._format_percentage(getattr(pitcher, "hard_pct", None)),
                "gb_pct": self._format_percentage(getattr(pitcher, "gb_pct", None)),
            }

        allowed, _ = game_state.is_game_action_allowed()
        if allowed:
            action_block_reason = None

        inning_scores = game_state.inning_scores
        scoreboard = {
            "away": list(inning_scores[0]),
            "home": list(inning_scores[1]),
        }

        max_innings = max(len(scoreboard["away"]), len(scoreboard["home"]), 9)
        situation = format_situation(game_state)

        offense_allowed = True
        defense_allowed = True
        progress_available = False
        if isinstance(control_state, dict):
            offense_allowed = bool(control_state.get("offense_allowed", True))
            defense_allowed = bool(control_state.get("defense_allowed", True))
            progress_available = bool(control_state.get("progress_available", False))

        last_play_raw = getattr(game_state, "last_play", None)
        if isinstance(last_play_raw, dict):
            raw_result = last_play_raw.get("result")
            raw_message = last_play_raw.get("message", "")
            raw_sequence = last_play_raw.get("sequence", 0)
            result_text = raw_result if raw_result is None else str(raw_result)
            if isinstance(raw_message, str):
                message_text = raw_message
            else:
                message_text = str(raw_message)
            if isinstance(raw_sequence, int):
                sequence_number = raw_sequence
            else:
                try:
                    sequence_number = int(raw_sequence)
                except (TypeError, ValueError):
                    sequence_number = getattr(game_state, "_play_sequence", 0)
        else:
            result_text = None
            message_text = ""
            sequence_number = getattr(game_state, "_play_sequence", 0)

        lineup_lookup = {}
        if batting_team and getattr(batting_team, "lineup", None):
            lineup_lookup = {id(player): index for index, player in enumerate(batting_team.lineup)}

        bases_payload = []
        for slot_index, runner in enumerate(game_state.bases):
            lineup_index = None
            if runner is not None and batting_team:
                lineup_index = lineup_lookup.get(id(runner))
                if lineup_index is None:
                    runner_name = getattr(runner, "name", None)
                    if runner_name:
                        for index, player in enumerate(batting_team.lineup):
                            if getattr(player, "name", None) == runner_name:
                                lineup_index = index
                                break

            bases_payload.append(
                {
                    "occupied": runner is not None,
                    "runner": getattr(runner, "name", None),
                    "speed": getattr(runner, "speed", None),
                    "speed_display": self._format_speed(getattr(runner, "speed", None))
                    if runner
                    else None,
                    "lineup_index": lineup_index,
                }
            )

        bunt_visible = not (
            game_state.can_squeeze() and not game_state.bases[0]
        )

        return (
            {
                "active": True,
                "inning": game_state.inning,
                "half": "top" if game_state.is_top_inning else "bottom",
                "half_label": "TOP" if game_state.is_top_inning else "BOTTOM",
                "outs": game_state.outs,
                "bases": bases_payload,
                "offense": offense,
                "defense": defense,
                "current_batter": current_batter,
                "current_pitcher": current_pitcher,
                "score": {
                    "home": game_state.home_score,
                    "away": game_state.away_score,
                },
                "hits": {
                    "home": count_hits(self._session.home_team),
                    "away": count_hits(self._session.away_team),
                },
                "errors": {"home": 0, "away": 0},
                "inning_scores": scoreboard,
                "actions": {
                    "swing": allowed and not game_state.game_ended and offense_allowed,
                    "bunt": allowed
                    and not game_state.game_ended
                    and offense_allowed
                    and game_state.can_bunt(),
                    "squeeze": allowed
                    and not game_state.game_ended
                    and offense_allowed
                    and game_state.can_squeeze(),
                    "steal": allowed
                    and not game_state.game_ended
                    and offense_allowed
                    and game_state.can_steal(),
                    "progress": progress_available and not game_state.game_ended,
                    "show_bunt": bunt_visible,
                },
                "action_block_reason": action_block_reason if not allowed else None,
                "defensive_errors": list(game_state.defensive_error_messages),
                "game_over": game_state.game_ended,
                "situation": situation,
                "matchup": format_matchup(current_batter, current_pitcher),
                "max_innings": max_innings,
                "last_play": {
                    "result": result_text,
                    "message": message_text,
                    "sequence": sequence_number,
                },
                "overlays": list(overlays or []),
                "control": control_state,
            },
            action_block_reason,
        )

    # ------------------------------------------------------------------
    # Team state
    # ------------------------------------------------------------------
    def _build_teams_state(self) -> Dict[str, Optional[Dict[str, object]]]:
        teams: Dict[str, Optional[Dict[str, object]]] = {"home": None, "away": None}
        control_state: Dict[str, object] = {}
        if hasattr(self._session, "get_control_state"):
            try:
                control_state = self._session.get_control_state()
            except Exception:
                control_state = {}

        mode_cpu = isinstance(control_state, dict) and control_state.get("mode") == "cpu"
        user_team_key = control_state.get("user_team") if isinstance(control_state, dict) else None
        cpu_team_key = control_state.get("cpu_team") if isinstance(control_state, dict) else None

        game_state = self._session.game_state

        for key, team in (("home", self._session.home_team), ("away", self._session.away_team)):
            if team is None:
                teams[key] = None
                continue

            lineup: List[Dict[str, object]] = []
            is_offense = bool(game_state and game_state.batting_team is team)
            current_batter_index = team.current_batter_index if team.lineup else 0
            for index, player in enumerate(team.lineup):
                summary = self._lineup_batter_summary(player)
                bats = getattr(player, "bats", None)
                bats_display = str(bats).upper() if bats else None
                fielding_raw = getattr(player, "fielding_skill", None)
                fielding_value = self._coerce_float(fielding_raw)
                lineup.append(
                    {
                        "index": index,
                        "order": index + 1,
                        "name": player.name,
                        "position": self._display_position(player),
                        "position_key": self._defensive_position_key(player),
                        "eligible": self._eligible_positions(player),
                        "eligible_all": self._eligible_positions_raw(player),
                        "pitcher_type": getattr(player, "pitcher_type", None),
                        "bats": bats_display,
                        "is_current_batter": is_offense and index == current_batter_index,
                        "fielding_rating": self._format_rating(fielding_raw),
                        "fielding_value": fielding_value,
                        "avg": summary["avg"],
                        "hr": summary["hr"],
                        "rbi": summary["rbi"],
                        "k_pct": self._format_percentage(getattr(player, "k_pct", None)),
                        "bb_pct": self._format_percentage(getattr(player, "bb_pct", None)),
                        "hard_pct": self._format_percentage(getattr(player, "hard_pct", None)),
                        "gb_pct": self._format_percentage(getattr(player, "gb_pct", None)),
                        "speed": self._format_speed(getattr(player, "speed", None)),
                    }
                )

            available_bench = (
                team.get_available_bench_players()
                if hasattr(team, "get_available_bench_players")
                else list(team.bench)
            )
            bench: List[Dict[str, object]] = []
            for bench_index, player in enumerate(available_bench):
                summary = self._lineup_batter_summary(player)
                bats = getattr(player, "bats", None)
                bats_display = str(bats).upper() if bats else None
                fielding_raw = getattr(player, "fielding_skill", None)
                fielding_value = self._coerce_float(fielding_raw)
                bench.append(
                    {
                        "index": bench_index,
                        "name": player.name,
                        "position": self._display_position(player),
                        "position_key": self._defensive_position_key(player),
                        "eligible": self._eligible_positions(player),
                        "eligible_all": self._eligible_positions_raw(player),
                        "pitcher_type": getattr(player, "pitcher_type", None),
                        "bats": bats_display,
                        "fielding_rating": self._format_rating(fielding_raw),
                        "fielding_value": fielding_value,
                        "avg": summary["avg"],
                        "hr": summary["hr"],
                        "rbi": summary["rbi"],
                    }
                )

            retired: List[Dict[str, object]] = []
            for player in getattr(team, "ejected_players", []) or []:
                fielding_raw = getattr(player, "fielding_skill", None)
                fielding_value = self._coerce_float(fielding_raw)
                retired.append(
                    {
                        "name": getattr(player, "name", "-"),
                        "position": self._display_position(player),
                        "position_key": self._defensive_position_key(player),
                        "eligible": self._eligible_positions(player),
                        "eligible_all": self._eligible_positions_raw(player),
                        "pitcher_type": getattr(player, "pitcher_type", None),
                        "fielding_rating": self._format_rating(fielding_raw),
                        "fielding_value": fielding_value,
                    }
                )

            pitchers: List[Dict[str, object]] = []
            seen_ids = set()
            if getattr(team, "current_pitcher", None):
                current = team.current_pitcher
                pitchers.append(self._serialize_pitcher(current, is_current=True))
                seen_ids.add(id(current))
            for pitcher in team.pitchers:
                if id(pitcher) in seen_ids:
                    continue
                pitchers.append(self._serialize_pitcher(pitcher, is_current=False))

            available_pitchers = (
                team.get_available_pitchers()
                if hasattr(team, "get_available_pitchers")
                else []
            )
            pitcher_options: List[Dict[str, object]] = []
            for pitcher_index, pitcher in enumerate(available_pitchers):
                pitcher_options.append(
                    {
                        "index": pitcher_index,
                        "name": pitcher.name,
                        "pitcher_type": getattr(pitcher, "pitcher_type", "P"),
                        "stamina": round(getattr(pitcher, "current_stamina", 0), 1),
                        "throws": getattr(pitcher, "throws", None),
                    }
                )

            teams[key] = {
                "name": team.name,
                "lineup": lineup,
                "bench": bench,
                "retired": retired,
                "pitchers": pitchers,
                "pitcher_options": pitcher_options,
                "stats": self._build_team_stats(team),
                "traits": self._build_team_traits(team),
                "controlled_by": (
                    "cpu" if mode_cpu and key == cpu_team_key else "user"
                ),
                "control_label": (
                    "CPU" if mode_cpu and key == cpu_team_key else "あなた"
                ),
            }

        return teams

    def _lineup_batter_summary(self, player) -> Dict[str, object]:
        if not player:
            return {"avg": ".000", "hr": 0, "rbi": 0}

        summary = self._serialize_batter_stats(player)
        avg = summary.get("avg") or ".000"
        hr = summary.get("hr")
        rbi = summary.get("rbi")

        return {
            "avg": avg,
            "hr": hr if hr is not None else 0,
            "rbi": rbi if rbi is not None else 0,
        }

    def _build_team_stats(self, team) -> Dict[str, List[Dict[str, object]]]:
        batting: List[Dict[str, object]] = []
        pitching: List[Dict[str, object]] = []

        if not team:
            return {"batting": batting, "pitching": pitching}

        seen_batters = set()

        def add_batter(player) -> None:
            player_id = id(player)
            if player_id in seen_batters:
                return
            seen_batters.add(player_id)
            position = getattr(player, "position", "") or ""
            if position.upper() == "P":
                return
            batting.append(self._serialize_batter_stats(player))

        for batter in getattr(team, "lineup", []):
            add_batter(batter)
        for batter in getattr(team, "bench", []):
            add_batter(batter)
        # Also include players who have left the game (e.g., via PH/PR)
        for batter in getattr(team, "ejected_players", []) or []:
            add_batter(batter)

        seen_pitchers = set()

        def add_pitcher(player) -> None:
            player_id = id(player)
            if player_id in seen_pitchers:
                return
            seen_pitchers.add(player_id)
            pitching.append(self._serialize_pitcher_stats(player))

        # Ensure active pitcher is included (relievers after a change)
        current_pitcher = getattr(team, "current_pitcher", None)
        if current_pitcher is not None:
            add_pitcher(current_pitcher)

        # Registered pitchers on the staff
        for pitcher in getattr(team, "pitchers", []):
            add_pitcher(pitcher)

        # Some data sources may tag pitchers in lineup/bench with position
        for pitcher in getattr(team, "lineup", []):
            if getattr(pitcher, "position", "").upper() == "P":
                add_pitcher(pitcher)
        for pitcher in getattr(team, "bench", []):
            if getattr(pitcher, "position", "").upper() == "P":
                add_pitcher(pitcher)

        # Previously used pitchers may be moved to ejected list; include them
        for player in getattr(team, "ejected_players", []) or []:
            if self._is_pitcher(player):
                add_pitcher(player)

        return {"batting": batting, "pitching": pitching}

    def _build_team_traits(self, team) -> Dict[str, List[Dict[str, object]]]:
        batting: List[Dict[str, object]] = []
        pitching: List[Dict[str, object]] = []

        if not team:
            return {"batting": batting, "pitching": pitching}

        for index, player in enumerate(getattr(team, "lineup", []) or []):
            batting.append(self._serialize_batter_traits(player, role_label=f"{index + 1}番"))

        available_bench = (
            team.get_available_bench_players()
            if hasattr(team, "get_available_bench_players")
            else list(getattr(team, "bench", []))
        )

        current_pitcher = getattr(team, "current_pitcher", None)
        seen_pitchers: set[int] = set()

        def add_pitcher(player, role_label: str) -> None:
            if not player:
                return
            player_id = id(player)
            if player_id in seen_pitchers:
                return
            seen_pitchers.add(player_id)
            pitching.append(self._serialize_pitcher_traits(player, role_label=role_label))

        if current_pitcher:
            add_pitcher(current_pitcher, "登板中")

        for idx, pitcher in enumerate(getattr(team, "pitchers", []) or []):
            if pitcher is current_pitcher:
                add_pitcher(pitcher, "登板中")
            else:
                add_pitcher(pitcher, f"控え{idx + 1}")

        for bench_index, player in enumerate(available_bench or []):
            role_label = f"ベンチ{bench_index + 1}"
            if self._is_pitcher(player):
                add_pitcher(player, role_label)
            else:
                batting.append(self._serialize_batter_traits(player, role_label=role_label))

        return {"batting": batting, "pitching": pitching}

    def _serialize_batter_traits(self, player, *, role_label: str) -> Dict[str, object]:
        if not player:
            return {
                "name": "-",
                "role_label": role_label,
                "position": "-",
                "bats": "-",
                "k_pct": "-",
                "bb_pct": "-",
                "hard_pct": "-",
                "gb_pct": "-",
                "speed": "-",
                "fielding": "-",
            }

        bats = getattr(player, "bats", None)
        bats_display = str(bats).upper() if bats else "-"

        return {
            "name": getattr(player, "name", "-"),
            "role_label": role_label,
            "position": self._display_position(player) or "-",
            "bats": bats_display,
            "k_pct": self._format_percentage(getattr(player, "k_pct", None)),
            "bb_pct": self._format_percentage(getattr(player, "bb_pct", None)),
            "hard_pct": self._format_percentage(getattr(player, "hard_pct", None)),
            "gb_pct": self._format_percentage(getattr(player, "gb_pct", None)),
            "speed": self._format_speed(getattr(player, "speed", None)),
            "fielding": self._format_rating(getattr(player, "fielding_skill", None)),
        }

    def _serialize_pitcher_traits(self, player, *, role_label: str) -> Dict[str, object]:
        if not player:
            return {
                "name": "-",
                "role_label": role_label,
                "pitcher_type": "-",
                "throws": "-",
                "k_pct": "-",
                "bb_pct": "-",
                "hard_pct": "-",
                "gb_pct": "-",
                "stamina": "-",
            }

        throws = getattr(player, "throws", None)
        throws_display = str(throws).upper() if throws else "-"
        pitcher_type = getattr(player, "pitcher_type", None)
        pitcher_type_display = str(pitcher_type).upper() if pitcher_type else (self._display_position(player) or "-")

        return {
            "name": getattr(player, "name", "-"),
            "role_label": role_label,
            "pitcher_type": pitcher_type_display,
            "throws": throws_display,
            "k_pct": self._format_percentage(getattr(player, "k_pct", None)),
            "bb_pct": self._format_percentage(getattr(player, "bb_pct", None)),
            "hard_pct": self._format_percentage(getattr(player, "hard_pct", None)),
            "gb_pct": self._format_percentage(getattr(player, "gb_pct", None)),
            "stamina": self._format_rating(getattr(player, "stamina", None)),
        }

    def _serialize_batter_stats(self, player) -> Dict[str, object]:
        stats = getattr(player, "stats", {}) or {}
        singles = stats.get("1B", 0)
        doubles = stats.get("2B", 0)
        triples = stats.get("3B", 0)
        homers = stats.get("HR", 0)
        hits = singles + doubles + triples + homers
        at_bats = stats.get("AB", 0)
        plate_appearances = stats.get("PA", at_bats + stats.get("BB", 0))

        if hasattr(player, "get_avg"):
            try:
                avg_value = float(player.get_avg())
            except Exception:  # pragma: no cover - defensive guard
                avg_value = StatsCalculator.calculate_batting_average(hits, at_bats)
        else:
            avg_value = StatsCalculator.calculate_batting_average(hits, at_bats)
        average = StatsCalculator.format_average(avg_value)

        return {
            "name": getattr(player, "name", "-"),
            "pa": plate_appearances,
            "ab": at_bats,
            "single": singles,
            "double": doubles,
            "triple": triples,
            "hr": homers,
            "h": hits,
            "runs": stats.get("R", 0),
            "rbi": stats.get("RBI", 0),
            "bb": stats.get("BB", 0),
            "sb": stats.get("SB", 0),
            "sba": stats.get("SBA", 0),
            "so": stats.get("SO", stats.get("K", 0)),
            "avg": average,
        }

    def _serialize_pitcher_stats(self, player) -> Dict[str, object]:
        raw_stats = getattr(player, "pitching_stats", {}) or {}
        innings = raw_stats.get("IP", 0)

        try:
            ip_display = StatsCalculator.format_inning_display(innings)
        except Exception:  # pragma: no cover - defensive guard
            ip_display = "0.0"

        if innings > 0 and hasattr(player, "get_era"):
            try:
                era_value = float(player.get_era())
            except Exception:  # pragma: no cover - defensive guard
                era_value = StatsCalculator.calculate_era(raw_stats.get("ER", 0), innings)
            era = StatsCalculator.format_average(era_value, 2)
        elif innings > 0:
            era = StatsCalculator.format_average(
                StatsCalculator.calculate_era(raw_stats.get("ER", 0), innings),
                2,
            )
        else:
            era = "-.--"

        if innings > 0 and hasattr(player, "get_whip"):
            try:
                whip_value = float(player.get_whip())
            except Exception:  # pragma: no cover - defensive guard
                whip_value = StatsCalculator.calculate_whip(
                    raw_stats.get("H", 0), raw_stats.get("BB", 0), innings
                )
            whip = StatsCalculator.format_average(whip_value, 2)
        elif innings > 0:
            whip = StatsCalculator.format_average(
                StatsCalculator.calculate_whip(
                    raw_stats.get("H", 0), raw_stats.get("BB", 0), innings
                ),
                2,
            )
        else:
            whip = "-.--"

        strikeouts = raw_stats.get("SO", raw_stats.get("K", 0))
        batters_faced = raw_stats.get("BF", 0)

        return {
            "name": getattr(player, "name", "-"),
            "ip": ip_display,
            "batters_faced": batters_faced,
            "h": raw_stats.get("H", 0),
            "r": raw_stats.get("R", 0),
            "er": raw_stats.get("ER", 0),
            "bb": raw_stats.get("BB", 0),
            "k": strikeouts,
            "so": strikeouts,
            "hr": raw_stats.get("HR", 0),
            "era": era,
            "whip": whip,
        }

    @staticmethod
    def _format_percentage(value) -> str:
        if value is None:
            return "-"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{numeric:.1f}%"

    @staticmethod
    def _format_speed(value) -> str:
        if value is None:
            return "-"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if math.isnan(numeric):
            return "-"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.1f}"

    @staticmethod
    def _format_rating(value) -> str:
        if value is None:
            return "-"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if math.isnan(numeric):
            return "-"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.1f}"

    @staticmethod
    def _coerce_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    @staticmethod
    def _is_pitcher(player) -> bool:
        if not player:
            return False
        if hasattr(player, "pitcher_type"):
            return True
        positions = getattr(player, "eligible_positions", None) or []
        return any(str(pos).upper() == "P" for pos in positions)

    @staticmethod
    def _serialize_pitcher(pitcher, is_current: bool) -> Dict[str, object]:
        # Attach quick stat snapshot for UI lineup display (ERA/IP/SO)
        raw_stats = getattr(pitcher, "pitching_stats", {}) or {}
        innings = raw_stats.get("IP", 0)
        # IP display
        try:
            ip_display = StatsCalculator.format_inning_display(innings)
        except Exception:
            ip_display = "0.0"
        # ERA display
        if innings and innings > 0:
            try:
                era_value = float(getattr(pitcher, "get_era")()) if hasattr(pitcher, "get_era") else None
            except Exception:
                era_value = None
            if era_value is None:
                try:
                    era_value = StatsCalculator.calculate_era(raw_stats.get("ER", 0), innings)
                except Exception:
                    era_value = 0.0
            era_display = StatsCalculator.format_average(era_value, 2)
        else:
            era_display = "-.--"
        # SO display
        strikeouts = raw_stats.get("SO", raw_stats.get("K", 0))

        return {
            "name": getattr(pitcher, "name", "-"),
            "stamina": round(getattr(pitcher, "current_stamina", 0), 1),
            "pitcher_type": getattr(pitcher, "pitcher_type", "P"),
            "throws": getattr(pitcher, "throws", None),
            "is_current": is_current,
            # Display-friendly stat fields
            "era": era_display,
            "ip": ip_display,
            "so": strikeouts,
        }

    @staticmethod
    def _eligible_positions(player) -> List[str]:
        if hasattr(player, "get_display_eligible_positions"):
            return list(player.get_display_eligible_positions())
        return list(getattr(player, "eligible_positions", []) or [])

    @staticmethod
    def _eligible_positions_raw(player) -> List[str]:
        positions = getattr(player, "eligible_positions", []) or []
        return [str(pos).upper() for pos in positions]

    @staticmethod
    def _defensive_position_key(player) -> Optional[str]:
        position = getattr(player, "current_position", None) or getattr(player, "position", None)
        if not position:
            return None
        position_key = str(position).upper()
        if position_key in {"SP", "RP"}:
            return "P"
        return position_key

    @staticmethod
    def _display_position(player) -> str:
        position = getattr(player, "current_position", None) or getattr(player, "position", "-")
        if (
            position
            and position.upper() == "P"
            and hasattr(player, "pitcher_type")
            and player.pitcher_type in {"SP", "RP"}
        ):
            return player.pitcher_type
        return position or "-"
