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

