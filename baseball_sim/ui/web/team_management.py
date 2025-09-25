"""Utilities for loading, persisting and validating web session teams."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from baseball_sim.data.loader import DataLoader
from baseball_sim.data.team_library import TeamLibraryError
from baseball_sim.gameplay.game import GameState

from .exceptions import GameSessionError
from .formatting import half_inning_banner


class TeamManagementMixin:
    """Provide team selection helpers for :class:`WebGameSession`."""

    home_team: Optional[object]
    away_team: Optional[object]
    _home_team_id: Optional[str]
    _away_team_id: Optional[str]
    _home_team_source: Optional[Dict[str, Any]]
    _away_team_source: Optional[Dict[str, Any]]

    def ensure_teams(
        self, force_reload: bool = False
    ) -> Tuple[Optional[object], Optional[object]]:
        """Load and cache the currently selected teams."""

        if force_reload or self.home_team is None or self.away_team is None:
            selection = self._team_library.ensure_selection_valid()
            self._home_team_id = selection.get("home")
            self._away_team_id = selection.get("away")

            if not self._home_team_id or not self._away_team_id:
                self._clear_team_cache()
                return self.home_team, self.away_team

            try:
                home_source = self._team_library.load_team(self._home_team_id)
                away_source = self._team_library.load_team(self._away_team_id)
            except TeamLibraryError as exc:
                self._clear_team_cache()
                self._notifications.publish("danger", str(exc))
                return self.home_team, self.away_team

            try:
                self.home_team, self.away_team = DataLoader.create_teams_from_data(
                    home_team_override=home_source,
                    away_team_override=away_source,
                )
                self._home_team_source = home_source
                self._away_team_source = away_source
            except Exception as exc:  # pragma: no cover - defensive
                self._clear_team_cache()
                self._notifications.publish(
                    "danger", f"Failed to load teams: {exc}"
                )
                return self.home_team, self.away_team

        return self.home_team, self.away_team

    def start_new_game(self, reload_teams: bool = False) -> Dict[str, Any]:
        """Create a new :class:`GameState` and reset bookkeeping."""

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
        """Exit the current game and return to the title screen."""

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
        """Force team data reload without starting a game."""

        self.ensure_teams(force_reload=True)
        self.game_state = None
        self._log.clear()
        self._game_over_announced = False
        self._action_block_reason = None
        self._notifications.publish("info", "Team data reloaded.")
        self._clear_simulation_results()
        return self.build_state()

    def get_team_library_state(self) -> Dict[str, object]:
        """Return metadata about team selection and readiness."""

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
        """Persist new team selection and reload as needed."""

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

    def get_team_definition(self, team_id: str) -> Dict[str, object]:
        """Fetch a team definition for editing."""

        if not team_id:
            raise GameSessionError("チームIDを指定してください。")
        try:
            return self._team_library.load_team(team_id)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

    def save_team_definition(
        self, team_id: Optional[str], team_data: Dict[str, object]
    ) -> Tuple[str, Dict[str, Any]]:
        """Create or update a team definition and return the session state."""

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
        """Delete a stored team definition and refresh state if required."""

        if not team_id:
            raise GameSessionError("チームIDを指定してください。")

        previously_active_ids = {self._home_team_id, self._away_team_id}

        try:
            selection = self._team_library.delete_team(team_id)
        except TeamLibraryError as exc:
            raise GameSessionError(str(exc)) from exc

        self._home_team_id = selection.get("home")
        self._away_team_id = selection.get("away")

        if team_id in previously_active_ids:
            self.ensure_teams(force_reload=True)
            self.game_state = None
            self._log.clear()
            self._game_over_announced = False
            self._action_block_reason = None
        else:
            self.ensure_teams(force_reload=False)

        self._notifications.publish("success", f"Team '{team_id}' を削除しました。")
        return self.build_state()

    # ------------------------------------------------------------------
    # Internal helpers
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
                "Cannot start game due to lineup validation issues: "
                + ", ".join(problems)
            )

    def _clear_team_cache(self) -> None:
        self.home_team = None
        self.away_team = None
        self._home_team_source = None
        self._away_team_source = None
        self._clear_simulation_results()

