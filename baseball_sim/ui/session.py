"""High level session object that coordinates the browser-facing game state."""

from __future__ import annotations

from typing import Any, Dict, Optional

from baseball_sim.config import setup_project_environment
from baseball_sim.data.team_library import TeamLibrary

from .exceptions import GameSessionError
from .gameplay_actions import GameplayActionsMixin
from .logging import SessionLog
from .notifications import NotificationCenter
from .simulation_controls import SimulationControlsMixin
from .state_builders import SessionStateBuilder
from .team_management import TeamManagementMixin

setup_project_environment()

__all__ = ["GameSessionError", "WebGameSession"]


class WebGameSession(
    SimulationControlsMixin, TeamManagementMixin, GameplayActionsMixin
):
    """Manage teams, game state and play log for the browser UI."""

    MAX_LOG_ENTRIES = 250

    def __init__(self) -> None:
        self.home_team = None
        self.away_team = None
        self.game_state = None
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
        self._control_mode: str = "manual"
        self._user_team_key: Optional[str] = None
        self._cpu_team_key: Optional[str] = None
        self.ensure_teams()

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
    # Control helpers for CPU vs player modes
    # ------------------------------------------------------------------
    def _configure_control_mode(
        self,
        mode: Optional[str] = None,
        user_team: Optional[str] = None,
    ) -> None:
        normalized_mode = "cpu" if str(mode or "").lower() == "cpu" else "manual"
        if normalized_mode != "cpu":
            self._control_mode = "manual"
            self._user_team_key = None
            self._cpu_team_key = None
            return

        normalized_team = "home" if str(user_team or "home").lower() == "home" else "away"
        self._control_mode = "cpu"
        self._user_team_key = normalized_team
        self._cpu_team_key = "away" if normalized_team == "home" else "home"

    def _get_team_by_key(self, key: Optional[str]):
        if key == "home":
            return self.home_team
        if key == "away":
            return self.away_team
        return None

    def _guard_offense_action(self) -> None:
        if not self._is_offense_action_allowed():
            raise GameSessionError(
                "守備中は攻撃操作を行えません。進行ボタンでCPUの攻撃を進めてください。"
            )

    def _guard_defense_action(self) -> None:
        if not self._is_defense_action_allowed():
            raise GameSessionError(
                "攻撃中は守備采配を行えません。攻撃が終了するまでお待ちください。"
            )

    def _guard_progress_action(self) -> None:
        if not self.is_progress_action_available():
            raise GameSessionError("進行操作は現在利用できません。")

    def _is_offense_action_allowed(self) -> bool:
        if self._control_mode != "cpu":
            return True
        if not self.game_state:
            return True
        user_team = self._get_team_by_key(self._user_team_key)
        return bool(user_team and self.game_state.batting_team is user_team)

    def _is_defense_action_allowed(self) -> bool:
        if self._control_mode != "cpu":
            return True
        if not self.game_state:
            return True
        user_team = self._get_team_by_key(self._user_team_key)
        return bool(user_team and self.game_state.fielding_team is user_team)

    def is_progress_action_available(self) -> bool:
        if self._control_mode != "cpu" or not self.game_state or self.game_state.game_ended:
            return False
        user_team = self._get_team_by_key(self._user_team_key)
        if not user_team or self.game_state.fielding_team is not user_team:
            return False
        allowed, _ = self.game_state.is_game_action_allowed()
        return bool(allowed)

    def get_control_state(self) -> Dict[str, Any]:
        mode = "cpu" if self._control_mode == "cpu" else "manual"
        user_team_key = self._user_team_key if mode == "cpu" else None
        cpu_team_key = self._cpu_team_key if mode == "cpu" else None

        def team_name(team_key: Optional[str]) -> Optional[str]:
            if team_key == "home":
                return getattr(self.home_team, "name", "Home")
            if team_key == "away":
                return getattr(self.away_team, "name", "Away")
            return None

        user_team_obj = self._get_team_by_key(user_team_key)
        cpu_team_obj = self._get_team_by_key(cpu_team_key)

        user_is_offense = True
        user_is_defense = True
        offense_allowed = True
        defense_allowed = True
        progress_available = False
        instruction = ""

        if mode == "cpu":
            if self.game_state:
                user_is_offense = self.game_state.batting_team is user_team_obj
                user_is_defense = self.game_state.fielding_team is user_team_obj
                offense_allowed = user_is_offense
                defense_allowed = user_is_defense
                allowed, reason = self.game_state.is_game_action_allowed()
                progress_available = (
                    not self.game_state.game_ended
                    and user_is_defense
                    and self.game_state.fielding_team is user_team_obj
                    and bool(allowed)
                )
                if not allowed and reason:
                    instruction = reason
                if not instruction:
                    if user_is_offense and not user_is_defense:
                        instruction = (
                            "攻撃中は守備采配を行えません。守備側の操作はCPUが担当します。"
                        )
                    elif user_is_defense and not user_is_offense:
                        instruction = "守備中は進行ボタンでCPUの攻撃を進めてください。"
            else:
                user_is_offense = True
                user_is_defense = True
                offense_allowed = True
                defense_allowed = True
                progress_available = False

        return {
            "mode": mode,
            "user_team": user_team_key,
            "cpu_team": cpu_team_key,
            "user_team_name": team_name(user_team_key),
            "cpu_team_name": team_name(cpu_team_key),
            "user_is_offense": user_is_offense,
            "user_is_defense": user_is_defense,
            "offense_allowed": offense_allowed,
            "defense_allowed": defense_allowed,
            "progress_available": progress_available,
            "instruction": instruction,
        }

