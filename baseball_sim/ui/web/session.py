"""Server-side helpers that expose the simulator to a browser client."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from baseball_sim.config import GameResults, setup_project_environment
from baseball_sim.data.loader import DataLoader
from baseball_sim.data.team_library import TeamLibrary, TeamLibraryError
from baseball_sim.gameplay.game import GameState
from baseball_sim.gameplay.substitutions import SubstitutionManager
from baseball_sim.interface.simulation import simulate_games

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
        self._team_library = TeamLibrary()
        self._team_library.ensure_initialized()
        selection = self._team_library.ensure_selection_valid()
        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")
        self._home_team_source: Optional[Dict[str, Any]] = None
        self._away_team_source: Optional[Dict[str, Any]] = None
        self._simulation_state: Dict[str, Any] = {
            "running": False,
            "last_run": None,
            "log": [],
            "default_games": 20,
            "limits": {"min_games": 1, "max_games": 200},
        }
        self.ensure_teams()

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def ensure_teams(
        self, force_reload: bool = False
    ) -> Tuple[Optional[object], Optional[object]]:
        """Load teams from disk if they are missing or a reload is requested."""

        if force_reload or self.home_team is None or self.away_team is None:
            selection = self._team_library.ensure_selection_valid()
            self._home_team_id = selection.get("home")
            self._away_team_id = selection.get("away")

            if not self._home_team_id or not self._away_team_id:
                self.home_team = None
                self.away_team = None
                self._home_team_source = None
                self._away_team_source = None
                self._clear_simulation_results()
                return self.home_team, self.away_team

            try:
                home_source = self._team_library.load_team(self._home_team_id)
                away_source = self._team_library.load_team(self._away_team_id)
            except TeamLibraryError as exc:
                self.home_team = None
                self.away_team = None
                self._home_team_source = None
                self._away_team_source = None
                self._clear_simulation_results()
                self._notifications.publish("danger", str(exc))
                return self.home_team, self.away_team

            try:
                self.home_team, self.away_team = DataLoader.create_teams_from_data(
                    home_team_override=home_source,
                    away_team_override=away_source,
                )
                self._home_team_source = home_source
                self._away_team_source = away_source
            except Exception as exc:
                self.home_team = None
                self.away_team = None
                self._home_team_source = None
                self._away_team_source = None
                self._clear_simulation_results()
                self._notifications.publish(
                    "danger", f"Failed to load teams: {exc}"
                )
                return self.home_team, self.away_team
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
        self._clear_simulation_results()
        return self.build_state()

    def get_team_library_state(self) -> Dict[str, object]:
        """Return metadata about available teams and current selection."""

        selection = self._team_library.ensure_selection_valid()
        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")
        state = self._team_library.describe()
        state["active"] = {
            "home": self._home_team_id,
            "away": self._away_team_id,
        }
        state["active_names"] = {
            "home": getattr(self.home_team, "name", None),
            "away": getattr(self.away_team, "name", None),
        }
        state["ready"] = bool(self.home_team and self.away_team) and state.get("ready", False)
        return state

    def update_team_selection(self, home_id: str, away_id: str) -> Dict[str, Any]:
        """Persist newly selected teams and reload data if necessary."""

        if not home_id or not away_id:
            raise GameSessionError("ホーム・アウェイ両方のチームを選択してください。")

        try:
            selection = self._team_library.set_selection(home_id, away_id)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")

        self.ensure_teams(force_reload=True)
        self.game_state = None
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None
        self._clear_simulation_results()

        if self.home_team and self.away_team:
            self._notifications.publish(
                "info",
                f"Teams selected: {self.away_team.name} @ {self.home_team.name}",
            )
        return self.build_state()

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _clear_simulation_results(self) -> None:
        self._simulation_state["last_run"] = None
        self._simulation_state["log"] = []
        self._simulation_state["running"] = False

    def get_simulation_state(self) -> Dict[str, Any]:
        state = {
            "enabled": bool(self.home_team and self.away_team),
            "running": bool(self._simulation_state.get("running", False)),
            "default_games": int(self._simulation_state.get("default_games", 20) or 20),
            "limits": self._simulation_state.get("limits", {"min_games": 1, "max_games": 200}),
            "last_run": self._simulation_state.get("last_run"),
            "log": list(self._simulation_state.get("log", []))[-20:],
        }
        return state

    def run_simulation(self, num_games: int) -> Dict[str, Any]:
        if num_games is None:
            raise GameSessionError("シミュレーションする試合数を指定してください。")

        limits = self._simulation_state.get("limits", {"min_games": 1, "max_games": 200})
        min_games = int(limits.get("min_games", 1) or 1)
        max_games = int(limits.get("max_games", 200) or 200)

        if num_games < min_games:
            raise GameSessionError(f"シミュレーション試合数は{min_games}以上で指定してください。")
        if num_games > max_games:
            raise GameSessionError(f"シミュレーション試合数は最大{max_games}試合までです。")

        self.ensure_teams()
        if not self.home_team or not self.away_team:
            raise GameSessionError("チームが読み込まれていません。チームを選択してください。")

        if not self._home_team_source or not self._away_team_source:
            raise GameSessionError("チームデータを読み込めませんでした。チーム選択を確認してください。")

        self._simulation_state["running"] = True

        progress_messages: List[str] = [f"Simulating {num_games} games..."]

        def handle_message(message: str) -> None:
            if not message:
                return
            progress_messages.append(str(message))

        try:
            results = simulate_games(
                num_games=num_games,
                output_file=None,
                progress_callback=None,
                message_callback=handle_message,
                home_team_data=self._home_team_source,
                away_team_data=self._away_team_source,
                save_to_file=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._simulation_state["running"] = False
            raise GameSessionError(f"シミュレーションに失敗しました: {exc}") from exc

        self._simulation_state["running"] = False
        self._simulation_state["default_games"] = num_games
        self._simulation_state["log"] = progress_messages[-20:]

        team_stats = results.get("team_stats") or {}

        def build_team_entry(team_key: str, team_obj) -> Dict[str, Any]:
            name = getattr(team_obj, "name", team_key.title())
            stats = team_stats.get(name, {})
            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            draws = int(stats.get("draws", 0))
            runs_scored = int(stats.get("runs_scored", 0))
            runs_allowed = int(stats.get("runs_allowed", 0))
            total = max(wins + losses + draws, 0)
            win_pct = wins / total if total else 0.0
            run_diff = runs_scored - runs_allowed
            return {
                "key": team_key,
                "name": name,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_pct": win_pct,
                "runs_scored": runs_scored,
                "runs_allowed": runs_allowed,
                "run_diff": run_diff,
            }

        teams_summary = [
            build_team_entry("away", self.away_team),
            build_team_entry("home", self.home_team),
        ]

        games = results.get("games") or []
        recent_games: List[Dict[str, Any]] = []
        for index, game in enumerate(games, start=1):
            try:
                home_score = int(game.get("home_score", 0))
                away_score = int(game.get("away_score", 0))
            except (TypeError, ValueError):  # pragma: no cover - fallback
                home_score, away_score = 0, 0

            winner: str
            if home_score > away_score:
                winner = "home"
            elif home_score < away_score:
                winner = "away"
            else:
                winner = "draw"

            recent_games.append(
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

        if recent_games:
            recent_games = recent_games[-5:]

        self._simulation_state["last_run"] = {
            "total_games": int(num_games),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "teams": teams_summary,
            "recent_games": recent_games,
        }

        self._notifications.publish(
            "success",
            f"{num_games}試合のシミュレーションが完了しました。",
        )

        return self.build_state()

    def get_team_definition(self, team_id: str) -> Dict[str, object]:
        """Fetch a stored team definition for editing."""

        if not team_id:
            raise GameSessionError("チームIDを指定してください。")
        try:
            return self._team_library.load_team(team_id)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

    def save_team_definition(
        self, team_id: Optional[str], team_data: Dict[str, object]
    ) -> Tuple[str, Dict[str, Any]]:
        """Create or update a team definition and return the new identifier."""

        try:
            saved_id = self._team_library.save_team(team_id, team_data)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

        selection = self._team_library.ensure_selection_valid()
        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")

        needs_reload = saved_id in {self._home_team_id, self._away_team_id}
        if needs_reload:
            self.ensure_teams(force_reload=True)
            self.game_state = None
            self._log.clear()
            self._game_over_announced = False
            self._action_block_reason = None

        team_name = str(team_data.get("name", saved_id))
        level = "info" if needs_reload else "success"
        self._notifications.publish(level, f"Team '{team_name}' saved.")

        state = self.build_state()
        return saved_id, state

    def delete_team_definition(self, team_id: str) -> Dict[str, Any]:
        """Delete a stored team definition and refresh state if needed."""
        if not team_id:
            raise GameSessionError("チームIDを指定してください。")

        # Capture whether the team is currently active before deletion
        previously_active_ids = {self._home_team_id, self._away_team_id}

        try:
            selection = self._team_library.delete_team(team_id)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

        # Update local cached selection
        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")

        # If the deleted team was active, force reload and reset game state
        if team_id in previously_active_ids:
            # ensure_selection_valid would already have adjusted IDs, but just in case
            self.ensure_teams(force_reload=True)
            self.game_state = None
            self._log.clear()
            self._game_over_announced = False
            self._action_block_reason = None
        else:
            # Even if not active, refresh description and keep things in sync
            self.ensure_teams(force_reload=False)

        self._notifications.publish("success", f"Team '{team_id}' を削除しました。")
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
        lineup_index = None
        try:
            lineup_index = batting_team.lineup.index(runner)
        except ValueError:
            runner_name = getattr(runner, "name", None)
            if runner_name:
                for idx, player in enumerate(batting_team.lineup):
                    if getattr(player, "name", None) == runner_name:
                        lineup_index = idx
                        break

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
            label = base_labels[base_index] if base_index < len(base_labels) else f"base {base_index + 1}"
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

