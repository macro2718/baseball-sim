"""Server-side helpers that expose the simulator to a browser client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from baseball_sim.config import GameResults, setup_project_environment
from baseball_sim.data.loader import DataLoader
from baseball_sim.gameplay.game import GameState

setup_project_environment()


class GameSessionError(RuntimeError):
    """Raised when an action cannot be performed in the current session state."""


@dataclass
class Notification:
    """Represents a one-off status message for the frontend."""

    level: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "message": self.message}


class WebGameSession:
    """Manage teams, game state and play log for the browser UI."""

    MAX_LOG_ENTRIES = 250

    def __init__(self) -> None:
        self.home_team = None
        self.away_team = None
        self.game_state: Optional[GameState] = None
        self._log: List[Dict[str, str]] = []
        self._game_over_announced = False
        self._action_block_reason: Optional[str] = None
        self._notification: Optional[Notification] = None
        self.ensure_teams()

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def ensure_teams(self, force_reload: bool = False) -> Tuple[Optional[object], Optional[object]]:
        """Load teams from disk if they are missing or a reload is requested."""

        if force_reload or self.home_team is None or self.away_team is None:
            self.home_team, self.away_team = DataLoader.create_teams_from_data()
        return self.home_team, self.away_team

    def start_new_game(self, reload_teams: bool = False) -> Dict[str, object]:
        """Create a fresh :class:`GameState` and reset session bookkeeping."""

        self.ensure_teams(force_reload=reload_teams)
        self._ensure_lineups_are_valid()

        if not self.home_team or not self.away_team:
            raise GameSessionError("Teams could not be loaded from the data files.")

        self.game_state = GameState(self.home_team, self.away_team)
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None
        self._notification = Notification(
            level="info",
            message=f"Game start: {self.away_team.name} at {self.home_team.name}",
        )
        self._append_log(self._notification.message, variant="info")
        self._append_log("=== TOP of the 1 ===", variant="highlight")
        return self.build_state()

    def stop_game(self) -> Dict[str, object]:
        """Return to the title screen without discarding loaded teams."""

        self.game_state = None
        self._game_over_announced = False
        self._action_block_reason = None
        self._notification = Notification(level="info", message="Game closed. Return to title screen.")
        return self.build_state()

    def clear_log(self) -> Dict[str, object]:
        """Remove all play log entries."""

        self._log.clear()
        self._notification = Notification(level="info", message="Play log cleared.")
        return self.build_state()

    def reload_teams(self) -> Dict[str, object]:
        """Force reloading team data from disk without starting a game."""

        self.ensure_teams(force_reload=True)
        self.game_state = None
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None
        self._notification = Notification(level="info", message="Team data reloaded.")
        return self.build_state()

    # ------------------------------------------------------------------
    # Gameplay actions
    # ------------------------------------------------------------------
    def execute_normal_play(self) -> Dict[str, object]:
        """Simulate a standard plate appearance."""

        if not self.game_state:
            raise GameSessionError("Game has not started yet.")

        if self.game_state.game_ended:
            self._action_block_reason = "The game has already ended."
            self._append_log(self._action_block_reason, variant="warning")
            return self.build_state()

        allowed, reason = self.game_state.is_game_action_allowed()
        if not allowed:
            self._action_block_reason = reason
            self._append_log(f"❌ {reason}", variant="danger")
            return self.build_state()

        self._action_block_reason = None

        batter = self.game_state.batting_team.current_batter
        pitcher = self.game_state.fielding_team.current_pitcher
        prev_inning = self.game_state.inning
        prev_half = self.game_state.is_top_inning

        result = self.game_state.calculate_result(batter, pitcher)
        message = self.game_state.apply_result(result, batter)

        self._append_log(f"{batter.name} vs {pitcher.name}", variant="header")
        variant = "success" if result in GameResults.POSITIVE_RESULTS else "danger"
        self._append_log(message, variant=variant)

        pitcher.decrease_stamina()

        inning_changed = (
            prev_inning != self.game_state.inning
            or prev_half != self.game_state.is_top_inning
        )
        if not inning_changed:
            self.game_state.batting_team.next_batter()
        else:
            self._append_log(self._half_inning_banner(), variant="highlight")

        if self.game_state.game_ended:
            self._record_game_over()

        return self.build_state()

    def execute_bunt(self) -> Dict[str, object]:
        """Attempt a bunt for the current batter."""

        if not self.game_state:
            raise GameSessionError("Game has not started yet.")

        if self.game_state.game_ended:
            self._action_block_reason = "The game has already ended."
            self._append_log(self._action_block_reason, variant="warning")
            return self.build_state()

        allowed, reason = self.game_state.is_game_action_allowed()
        if not allowed:
            self._action_block_reason = reason
            self._append_log(f"❌ {reason}", variant="danger")
            return self.build_state()

        if not self.game_state.can_bunt():
            self._action_block_reason = "Bunt not allowed (need runners on base and fewer than 2 outs)."
            self._append_log(self._action_block_reason, variant="warning")
            return self.build_state()

        self._action_block_reason = None

        batter = self.game_state.batting_team.current_batter
        pitcher = self.game_state.fielding_team.current_pitcher
        prev_inning = self.game_state.inning
        prev_half = self.game_state.is_top_inning

        result_message = self.game_state.execute_bunt(batter, pitcher)

        if "Cannot bunt" in result_message or "バントはできません" in result_message:
            self._action_block_reason = result_message
            self._append_log(result_message, variant="warning")
            return self.build_state()

        self._append_log(f"{batter.name} attempts a bunt against {pitcher.name}", variant="header")
        self._append_log(result_message, variant="success")

        pitcher.decrease_stamina()
        self.game_state.batting_team.next_batter()

        inning_changed = (
            prev_inning != self.game_state.inning
            or prev_half != self.game_state.is_top_inning
        )
        if inning_changed:
            self._append_log(self._half_inning_banner(), variant="highlight")

        if self.game_state.game_ended:
            self._record_game_over()
        elif (
            self.game_state.inning >= 9
            and not self.game_state.is_top_inning
            and self.game_state.home_score > self.game_state.away_score
        ):
            self._record_game_over()

        return self.build_state()

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------
    def build_state(self) -> Dict[str, object]:
        """Return a dictionary representing the entire UI state."""

        title = self._build_title_state()
        teams = self._build_teams_state()
        game = self._build_game_state(teams)

        payload = {
            "title": title,
            "teams": teams,
            "game": game,
            "log": list(self._log),
        }
        if self._notification:
            payload["notification"] = self._notification.to_dict()
        else:
            payload["notification"] = None
        self._notification = None
        return payload

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _ensure_lineups_are_valid(self) -> None:
        title_state = self._build_title_state()
        if not title_state["ready"]:
            problematic = [
                f"{info['name']} ({len(info['errors'])} issue(s))"
                for info in (title_state["home"], title_state["away"])
                if info and not info["valid"]
            ]
            raise GameSessionError(
                "Cannot start game due to lineup validation issues: " + ", ".join(problematic)
            )

    def _append_log(self, message: str, variant: str = "info") -> None:
        if not message:
            return
        self._log.append({"text": message, "variant": variant})
        if len(self._log) > self.MAX_LOG_ENTRIES:
            del self._log[:-self.MAX_LOG_ENTRIES]

    def _record_game_over(self) -> None:
        if self._game_over_announced or not self.game_state:
            return

        home_score = self.game_state.home_score
        away_score = self.game_state.away_score
        home_name = self.home_team.name if self.home_team else "Home"
        away_name = self.away_team.name if self.away_team else "Away"

        self._append_log("Game Over", variant="info")
        self._append_log(
            f"Final Score: {away_name} {away_score} - {home_name} {home_score}",
            variant="info",
        )
        if home_score > away_score:
            self._append_log(f"{home_name} win!", variant="success")
        elif away_score > home_score:
            self._append_log(f"{away_name} win!", variant="success")
        else:
            self._append_log("Game ends in a tie.", variant="warning")

        self._game_over_announced = True
        self._notification = Notification(level="info", message="Game finished.")

    def _half_inning_banner(self) -> str:
        if not self.game_state:
            return ""
        half = "TOP" if self.game_state.is_top_inning else "BOTTOM"
        return f"=== {half} of the {self.game_state.inning} ==="

    def _build_title_state(self) -> Dict[str, object]:
        home_status = self._team_status(self.home_team)
        away_status = self._team_status(self.away_team)
        ready = bool(home_status["valid"] and away_status["valid"])

        if not self.home_team or not self.away_team:
            hint = "Teams could not be loaded. Check data files."
        elif ready:
            hint = "Lineups ready. Press Start Game to begin."
        else:
            issues = []
            if not home_status["valid"]:
                issues.append(f"{home_status['name']}: {len(home_status['errors'])} issue(s)")
            if not away_status["valid"]:
                issues.append(f"{away_status['name']}: {len(away_status['errors'])} issue(s)")
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

    def _build_game_state(self, teams: Dict[str, Optional[Dict[str, object]]]) -> Dict[str, object]:
        if not self.game_state:
            return {
                "active": False,
                "actions": {"swing": False, "bunt": False},
                "action_block_reason": self._action_block_reason,
                "game_over": False,
                "defensive_errors": [],
                "score": {"home": 0, "away": 0},
                "inning_scores": {"home": [], "away": []},
                "situation": "Waiting for a new game.",
            }

        batting_team = self.game_state.batting_team
        offense = None
        defense = None
        if batting_team is self.home_team:
            offense, defense = "home", "away"
        elif batting_team is self.away_team:
            offense, defense = "away", "home"

        current_batter = None
        if batting_team and batting_team.lineup:
            batter = batting_team.current_batter
            current_batter = {
                "name": batter.name,
                "order": batting_team.current_batter_index + 1,
                "position": self._display_position(batter),
            }

        current_pitcher = None
        if self.game_state.fielding_team and self.game_state.fielding_team.current_pitcher:
            pitcher = self.game_state.fielding_team.current_pitcher
            current_pitcher = {
                "name": pitcher.name,
                "stamina": round(getattr(pitcher, "current_stamina", 0), 1),
                "pitcher_type": getattr(pitcher, "pitcher_type", "P"),
            }

        allowed, reason = self.game_state.is_game_action_allowed()
        if allowed:
            self._action_block_reason = None

        inning_scores = self.game_state.inning_scores
        scoreboard = {
            "away": list(inning_scores[0]),
            "home": list(inning_scores[1]),
        }

        situation = self._format_situation()

        return {
            "active": True,
            "inning": self.game_state.inning,
            "half": "top" if self.game_state.is_top_inning else "bottom",
            "half_label": "TOP" if self.game_state.is_top_inning else "BOTTOM",
            "outs": self.game_state.outs,
            "bases": [
                {"occupied": runner is not None, "runner": getattr(runner, "name", None)}
                for runner in self.game_state.bases
            ],
            "offense": offense,
            "defense": defense,
            "current_batter": current_batter,
            "current_pitcher": current_pitcher,
            "score": {"home": self.game_state.home_score, "away": self.game_state.away_score},
            "inning_scores": scoreboard,
            "actions": {
                "swing": allowed and not self.game_state.game_ended,
                "bunt": allowed and not self.game_state.game_ended and self.game_state.can_bunt(),
            },
            "action_block_reason": self._action_block_reason if not allowed else None,
            "defensive_errors": list(self.game_state.defensive_error_messages),
            "game_over": self.game_state.game_ended,
            "situation": situation,
            "matchup": self._format_matchup(current_batter, current_pitcher),
        }

    def _build_teams_state(self) -> Dict[str, Optional[Dict[str, object]]]:
        teams: Dict[str, Optional[Dict[str, object]]] = {"home": None, "away": None}
        for key, team in (("home", self.home_team), ("away", self.away_team)):
            if team is None:
                teams[key] = None
                continue

            lineup = []
            is_offense = bool(self.game_state and self.game_state.batting_team is team)
            current_batter_index = team.current_batter_index if team.lineup else 0
            for index, player in enumerate(team.lineup):
                lineup.append(
                    {
                        "order": index + 1,
                        "name": player.name,
                        "position": self._display_position(player),
                        "eligible": self._eligible_positions(player),
                        "is_current_batter": is_offense and index == current_batter_index,
                    }
                )

            bench = [
                {
                    "name": player.name,
                    "eligible": self._eligible_positions(player),
                }
                for player in team.bench
            ]

            pitchers = []
            seen_ids = set()
            if getattr(team, "current_pitcher", None):
                current = team.current_pitcher
                pitchers.append(self._serialize_pitcher(current, is_current=True))
                seen_ids.add(id(current))
            for pitcher in team.pitchers:
                if id(pitcher) in seen_ids:
                    continue
                pitchers.append(self._serialize_pitcher(pitcher, is_current=False))

            teams[key] = {
                "name": team.name,
                "lineup": lineup,
                "bench": bench,
                "pitchers": pitchers,
            }

        return teams

    def _serialize_pitcher(self, pitcher, is_current: bool) -> Dict[str, object]:
        return {
            "name": pitcher.name,
            "stamina": round(getattr(pitcher, "current_stamina", 0), 1),
            "pitcher_type": getattr(pitcher, "pitcher_type", "P"),
            "is_current": is_current,
        }

    def _eligible_positions(self, player) -> List[str]:
        if hasattr(player, "get_display_eligible_positions"):
            return list(player.get_display_eligible_positions())
        return list(getattr(player, "eligible_positions", []) or [])

    def _display_position(self, player) -> str:
        position = getattr(player, "current_position", None) or getattr(player, "position", "-")
        if (
            position
            and position.upper() == "P"
            and hasattr(player, "pitcher_type")
            and player.pitcher_type in {"SP", "RP"}
        ):
            return player.pitcher_type
        return position or "-"

    def _format_situation(self) -> str:
        if not self.game_state:
            return ""

        runners = []
        if self.game_state.bases[0]:
            runners.append("1st")
        if self.game_state.bases[1]:
            runners.append("2nd")
        if self.game_state.bases[2]:
            runners.append("3rd")

        runner_text = "Bases Empty" if not runners else f"Runners on {', '.join(runners)}"
        half = "Top" if self.game_state.is_top_inning else "Bottom"
        return f"{half} {self.game_state.inning} — {self.game_state.outs} Outs — {runner_text}"

    @staticmethod
    def _format_matchup(batter: Optional[Dict[str, object]], pitcher: Optional[Dict[str, object]]) -> Optional[str]:
        if not batter or not pitcher:
            return None
        return f"{batter['name']} vs {pitcher['name']}"
