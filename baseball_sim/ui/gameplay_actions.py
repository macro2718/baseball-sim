"""Gameplay interaction helpers extracted from :mod:`session`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from baseball_sim.config import GameResults
from baseball_sim.gameplay.substitutions import SubstitutionManager

from .exceptions import GameSessionError
from .formatting import half_inning_banner


class GameplayActionsMixin:
    """Encapsulate the command handlers used by the browser UI."""

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
            self._publish_positive_result(result, batter)
        else:
            self._publish_negative_result(result, batter)

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

    def execute_steal(self) -> Dict[str, Any]:
        """Attempt a steal with the current base runners."""

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

        if not self.game_state.can_steal():
            message = "盗塁はできません（適切な走者がいません）。"
            self._action_block_reason = message
            self._log.append(message, variant="warning")
            return self.build_state()

        self._action_block_reason = None

        prev_inning = self.game_state.inning
        prev_half = self.game_state.is_top_inning

        self._log.append("🔐 盗塁指示", variant="header")

        result_info = self.game_state.execute_steal()
        result_key = result_info.get("result")
        message = result_info.get("message", "盗塁を試みました。")
        success = bool(result_info.get("success"))

        if result_key == GameResults.STEAL_NOT_ALLOWED:
            self._action_block_reason = message
            self._log.append(message, variant="warning")
            return self.build_state()

        variant = "success" if success else "danger"
        self._log.append(message, variant=variant)
        notification_type = "success" if success else "danger"
        self._notifications.publish(notification_type, message)

        inning_changed = (
            prev_inning != self.game_state.inning
            or prev_half != self.game_state.is_top_inning
        )
        if inning_changed:
            banner = half_inning_banner(self.game_state, self.home_team, self.away_team)
            self._log.extend_banner(banner)

        if self.game_state.game_ended:
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

    def execute_pinch_run(self, base_index: int, bench_index: int) -> Dict[str, Any]:
        """Replace an active base runner with a bench player."""

        if not self.game_state or not self.game_state.batting_team:
            raise GameSessionError("Game has not started yet.")

        bases = self.game_state.bases
        total_bases = len(bases)
        if base_index < 0 or base_index >= total_bases:
            raise GameSessionError("Invalid base index for pinch run request.")

        runner = bases[base_index]
        if runner is None:
            raise GameSessionError("The selected base is not occupied.")

        batting_team = self.game_state.batting_team
        lineup_index = _find_lineup_index(batting_team.lineup, runner)

        if lineup_index is None:
            raise GameSessionError("Could not match the selected runner to the lineup.")

        substitution_manager = SubstitutionManager(batting_team)
        success, result_message = substitution_manager.execute_defensive_substitution(
            bench_index, lineup_index
        )

        original_name = getattr(runner, "name", "runner")
        message = result_message
        if success:
            new_runner = batting_team.lineup[lineup_index]
            bases[base_index] = new_runner
            base_labels = ["first base", "second base", "third base"]
            label = (
                base_labels[base_index]
                if base_index < len(base_labels)
                else f"base {base_index + 1}"
            )
            message = (
                f"{new_runner.name} pinch runs for {original_name} on {label}. {result_message}"
            )

        self._notifications.publish("success" if success else "danger", message)
        variant = "highlight" if success else "danger"
        self._log.append(message, variant=variant)
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
    # Helper methods
    # ------------------------------------------------------------------
    def _publish_positive_result(self, result: str, batter) -> None:
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

    def _publish_negative_result(self, result: str, batter) -> None:
        if result == GameResults.STRIKEOUT:
            self._notifications.publish("warning", f"⚾ {batter.name} strikes out")
        else:
            self._notifications.publish("info", f"{batter.name}: {result}")

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


def _find_lineup_index(lineup: List[object], runner: object) -> Optional[int]:
    try:
        return lineup.index(runner)
    except ValueError:
        runner_name = getattr(runner, "name", None)
        if runner_name:
            for idx, player in enumerate(lineup):
                if getattr(player, "name", None) == runner_name:
                    return idx
    return None

