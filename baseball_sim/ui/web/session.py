"""Server-side helpers that expose the simulator to a browser client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from baseball_sim.config import GameResults, setup_project_environment
from baseball_sim.data.loader import DataLoader
from baseball_sim.gameplay.game import GameState
from baseball_sim.gameplay.substitutions import SubstitutionManager

from .formatting import half_inning_banner
from .logging import SessionLog
from .notifications import NotificationCenter
from .state_builders import SessionStateBuilder

setup_project_environment()


class GameSessionError(RuntimeError):
    """Raised when an action cannot be performed in the current session state."""


class WebGameSession:
    """Manage teams, game state and play log for the browser UI."""

    MAX_LOG_ENTRIES = 250

    def __init__(self) -> None:
        self.home_team = None
        self.away_team = None
        self.game_state: Optional[GameState] = None
        self._log = SessionLog(self.MAX_LOG_ENTRIES)
        self._notifications = NotificationCenter()
        self._state_builder = SessionStateBuilder(self)
        self._game_over_announced = False
        self._action_block_reason: Optional[str] = None
        self.ensure_teams()

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def ensure_teams(
        self, force_reload: bool = False
    ) -> Tuple[Optional[object], Optional[object]]:
        """Load teams from disk if they are missing or a reload is requested."""

        if force_reload or self.home_team is None or self.away_team is None:
            self.home_team, self.away_team = DataLoader.create_teams_from_data()
        return self.home_team, self.away_team

    def start_new_game(self, reload_teams: bool = False) -> Dict[str, Any]:
        """Create a fresh :class:`GameState` and reset session bookkeeping."""

        self.ensure_teams(force_reload=reload_teams)
        self._ensure_lineups_are_valid()

        if not self.home_team or not self.away_team:
            raise GameSessionError("Teams could not be loaded from the data files.")

        self.game_state = GameState(self.home_team, self.away_team)
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None

        self._notifications.publish(
            "info", f"New game started: {self.away_team.name} @ {self.home_team.name}"
        )
        self._log.append("=" * 60, variant="highlight")
        self._log.append("🏟️  NEW GAME STARTED  🏟️", variant="success")
        self._log.append("=" * 60, variant="highlight")
        self._log.append(
            f"⚾ {self.away_team.name} (Away) @ {self.home_team.name} (Home)", variant="info"
        )
        self._log.append("📅 Starting at inning 1", variant="info")
        self._log.append("=" * 60, variant="highlight")
        banner = half_inning_banner(self.game_state, self.home_team, self.away_team)
        self._log.extend_banner(banner)
        return self.build_state()

    def stop_game(self) -> Dict[str, Any]:
        """Return to the title screen without discarding loaded teams."""

        self.game_state = None
        self._game_over_announced = False
        self._action_block_reason = None
        self._notifications.publish("info", "Game closed. Return to title screen.")
        return self.build_state()

    def clear_log(self) -> Dict[str, Any]:
        """Remove all play log entries."""

        self._log.clear()
        self._notifications.publish("info", "Play log cleared.")
        return self.build_state()

    def reload_teams(self) -> Dict[str, Any]:
        """Force reloading team data from disk without starting a game."""

        self.ensure_teams(force_reload=True)
        self.game_state = None
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None
        self._notifications.publish("info", "Team data reloaded.")
        return self.build_state()

    # ------------------------------------------------------------------
    # Gameplay actions
    # ------------------------------------------------------------------
    def execute_normal_play(self) -> Dict[str, Any]:
        """Simulate a standard plate appearance."""

        if not self.game_state:
            raise GameSessionError("Game has not started yet.")

        if self.game_state.game_ended:
            self._action_block_reason = "The game has already ended."
            self._log.append(self._action_block_reason, variant="warning")
            return self.build_state()

        allowed, reason = self.game_state.is_game_action_allowed()
        if not allowed:
            self._action_block_reason = reason
            self._log.append(f"❌ {reason}", variant="danger")
            return self.build_state()

        self._action_block_reason = None

        batter = self.game_state.batting_team.current_batter
        pitcher = self.game_state.fielding_team.current_pitcher
        prev_inning = self.game_state.inning
        prev_half = self.game_state.is_top_inning

        result = self.game_state.calculate_result(batter, pitcher)
        message = self.game_state.apply_result(result, batter)

        self._log.append(f"{batter.name} vs {pitcher.name}", variant="header")
        variant = "success" if result in GameResults.POSITIVE_RESULTS else "danger"
        self._log.append(message, variant=variant)

        if result in GameResults.POSITIVE_RESULTS:
            if result == GameResults.HOME_RUN:
                self._notifications.publish("success", f"🚀 {batter.name} hits a HOME RUN!")
            elif result == GameResults.TRIPLE:
                self._notifications.publish("success", f"⚡ {batter.name} hits a TRIPLE!")
            elif result == GameResults.DOUBLE:
                self._notifications.publish("success", f"💨 {batter.name} hits a DOUBLE!")
            elif result == GameResults.SINGLE:
                self._notifications.publish("success", f"✅ {batter.name} gets a hit!")
            elif result == GameResults.WALK:
                self._notifications.publish("info", f"🚶 {batter.name} draws a walk")
        else:
            if result == GameResults.STRIKEOUT:
                self._notifications.publish("warning", f"⚾ {batter.name} strikes out")
            else:
                self._notifications.publish("info", f"{batter.name}: {result}")

        pitcher.decrease_stamina()

        inning_changed = (
            prev_inning != self.game_state.inning
            or prev_half != self.game_state.is_top_inning
        )
        if not inning_changed:
            self.game_state.batting_team.next_batter()
        else:
            banner = half_inning_banner(self.game_state, self.home_team, self.away_team)
            self._log.extend_banner(banner)

        if self.game_state.game_ended:
            self._record_game_over()

        return self.build_state()

    def execute_bunt(self) -> Dict[str, Any]:
        """Attempt a bunt for the current batter."""

        if not self.game_state:
            raise GameSessionError("Game has not started yet.")

        if self.game_state.game_ended:
            self._action_block_reason = "The game has already ended."
            self._log.append(self._action_block_reason, variant="warning")
            return self.build_state()

        allowed, reason = self.game_state.is_game_action_allowed()
        if not allowed:
            self._action_block_reason = reason
            self._log.append(f"❌ {reason}", variant="danger")
            return self.build_state()

        if not self.game_state.can_bunt():
            self._action_block_reason = (
                "Bunt not allowed (need runners on base and fewer than 2 outs)."
            )
            self._log.append(self._action_block_reason, variant="warning")
            return self.build_state()

        self._action_block_reason = None

        batter = self.game_state.batting_team.current_batter
        pitcher = self.game_state.fielding_team.current_pitcher
        prev_inning = self.game_state.inning
        prev_half = self.game_state.is_top_inning

        result_message = self.game_state.execute_bunt(batter, pitcher)

        if "Cannot bunt" in result_message or "バントはできません" in result_message:
            self._action_block_reason = result_message
            self._log.append(result_message, variant="warning")
            return self.build_state()

        self._log.append(
            f"{batter.name} attempts a bunt against {pitcher.name}", variant="header"
        )

        if "Cannot bunt" not in result_message and "バントはできません" not in result_message:
            self._log.append(result_message, variant="success")
            self._notifications.publish("success", f"🏃 {batter.name} executes a bunt!")
        else:
            self._log.append(result_message, variant="warning")
            self._notifications.publish("warning", f"❌ {batter.name}'s bunt attempt fails")

        pitcher.decrease_stamina()
        self.game_state.batting_team.next_batter()

        inning_changed = (
            prev_inning != self.game_state.inning
            or prev_half != self.game_state.is_top_inning
        )
        if inning_changed:
            banner = half_inning_banner(self.game_state, self.home_team, self.away_team)
            self._log.extend_banner(banner)

        if self.game_state.game_ended:
            self._record_game_over()
        elif (
            self.game_state.inning >= 9
            and not self.game_state.is_top_inning
            and self.game_state.home_score > self.game_state.away_score
        ):
            self._record_game_over()

        return self.build_state()

    def execute_pinch_hit(self, lineup_index: int, bench_index: int) -> Dict[str, Any]:
        """Replace the selected batter with a bench player."""

        if not self.game_state or not self.game_state.batting_team:
            raise GameSessionError("Game has not started yet.")

        substitution_manager = SubstitutionManager(self.game_state.batting_team)
        success, message = substitution_manager.execute_pinch_hit(bench_index, lineup_index)

        self._notifications.publish("success" if success else "danger", message)
        self._log.append(message, variant="highlight" if success else "danger")
        return self.build_state()

    def execute_defensive_substitution(
        self,
        *,
        lineup_index: Optional[int] = None,
        bench_index: Optional[int] = None,
        swaps: Optional[List[Dict[str, Any]]] = None,
        force_illegal: bool = False,
    ) -> Dict[str, object]:
        """Swap defensive players according to the provided instruction."""

        if not self.game_state or not self.game_state.fielding_team:
            raise GameSessionError("Game has not started yet.")

        substitution_manager = SubstitutionManager(self.game_state.fielding_team)

        if swaps is not None:
            success, message = substitution_manager.execute_defensive_plan(
                swaps, allow_illegal=force_illegal
            )
        else:
            if lineup_index is None or bench_index is None:
                raise GameSessionError("Invalid defensive substitution request.")
            success, message = substitution_manager.execute_defensive_substitution(
                bench_index, lineup_index, allow_illegal=force_illegal
            )

        self._notifications.publish("success" if success else "danger", message)
        variant = "highlight" if success else "danger"
        self._log.append(message, variant=variant)
        if success:
            self._refresh_defense_status()
        return self.build_state()

    def execute_pitcher_change(self, pitcher_index: int) -> Dict[str, object]:
        """Bring in a new pitcher for the fielding team."""

        if not self.game_state or not self.game_state.fielding_team:
            raise GameSessionError("Game has not started yet.")

        substitution_manager = SubstitutionManager(self.game_state.fielding_team)
        success, message = substitution_manager.execute_pitcher_change(pitcher_index)

        self._notifications.publish("success" if success else "danger", message)
        variant = "highlight" if success else "danger"
        self._log.append(message, variant=variant)
        if success:
            self._refresh_defense_status()
        return self.build_state()

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------
    def build_state(self) -> Dict[str, Any]:
        """Return a dictionary representing the entire UI state."""

        notification = self._notifications.consume()
        payload, updated_reason = self._state_builder.build_state(
            log_entries=self._log.as_list(),
            notification=notification,
            action_block_reason=self._action_block_reason,
        )
        self._action_block_reason = updated_reason
        return payload

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _ensure_lineups_are_valid(self) -> None:
        problems = []
        for team in (self.home_team, self.away_team):
            if not team:
                problems.append("Team not loaded")
                continue
            errors = team.validate_lineup()
            if errors:
                problems.append(f"{team.name} ({len(errors)} issue(s))")
        if problems:
            raise GameSessionError(
                "Cannot start game due to lineup validation issues: " + ", ".join(problems)
            )

    def _refresh_defense_status(self) -> None:
        if not self.game_state:
            return
        evaluate = getattr(self.game_state, "_evaluate_defensive_alignment", None)
        if callable(evaluate):
            evaluate()

    def _record_game_over(self) -> None:
        if self._game_over_announced or not self.game_state:
            return

        home_score = self.game_state.home_score
        away_score = self.game_state.away_score
        home_name = self.home_team.name if self.home_team else "Home"
        away_name = self.away_team.name if self.away_team else "Away"

        self._log.append("=" * 50, variant="highlight")
        self._log.append("GAME OVER", variant="info")
        self._log.append("=" * 50, variant="highlight")
        self._log.append(
            f"Final Score: {away_name} {away_score} - {home_name} {home_score}",
            variant="info",
        )

        if home_score > away_score:
            winner_msg = f"🏆 {home_name} WINS!"
            winner_detail = (
                f"{home_name} defeats {away_name} by {home_score - away_score} run(s)"
            )
            self._log.append(winner_msg, variant="success")
            self._log.append(winner_detail, variant="success")
            notification_msg = f"Game finished. {home_name} wins {home_score}-{away_score}!"
        elif away_score > home_score:
            winner_msg = f"🏆 {away_name} WINS!"
            winner_detail = (
                f"{away_name} defeats {home_name} by {away_score - home_score} run(s)"
            )
            self._log.append(winner_msg, variant="success")
            self._log.append(winner_detail, variant="success")
            notification_msg = f"Game finished. {away_name} wins {away_score}-{home_score}!"
        else:
            tie_msg = "Game ends in a tie."
            self._log.append(tie_msg, variant="warning")
            notification_msg = f"Game finished in a tie {home_score}-{away_score}."

        innings_played = self.game_state.inning
        if not self.game_state.is_top_inning and innings_played >= 9:
            innings_msg = f"Game completed in {innings_played} innings"
        else:
            suffix = (
                "st"
                if innings_played == 1
                else "nd"
                if innings_played == 2
                else "rd"
                if innings_played == 3
                else "th"
            )
            innings_msg = f"Game ended in the {innings_played}{suffix} inning"
        self._log.append(innings_msg, variant="info")

        self._log.append("=" * 50, variant="highlight")

        self._game_over_announced = True
        self._notifications.publish("success", notification_msg)

