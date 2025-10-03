"""Utilities for loading, persisting and validating web session teams."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from baseball_sim.data.loader import DataLoader
from baseball_sim.data.team_library import TeamLibraryError
from baseball_sim.gameplay.game import GameState
from baseball_sim.infrastructure.logging_utils import logger as root_logger

from .exceptions import GameSessionError
from .formatting import half_inning_banner


LOGGER = root_logger.getChild("ui.team_management")


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
                self.home_team, self.away_team, warnings = DataLoader.create_teams_from_data(
                    home_team_override=home_source,
                    away_team_override=away_source,
                )
                self._home_team_source = home_source
                self._away_team_source = away_source
                if warnings:
                    for warning in warnings:
                        LOGGER.warning("Team setup warning: %s", warning)
                        self._notifications.publish("warning", warning)
                        play_log = getattr(self, "_log", None)
                        append = getattr(play_log, "append", None)
                        if callable(append):
                            append(f"⚠️ {warning}", variant="warning")
            except Exception as exc:  # pragma: no cover - defensive
                self._clear_team_cache()
                self._notifications.publish(
                    "danger", f"Failed to load teams: {exc}"
                )
                return self.home_team, self.away_team

        return self.home_team, self.away_team

    def _prepare_game_setup(
        self, control_mode: Optional[str], user_team: Optional[str]
    ) -> None:
        if hasattr(self, "_configure_control_mode"):
            try:
                self._configure_control_mode(control_mode, user_team)
            except Exception:
                # Fallback to manual control if configuration fails unexpectedly
                self._configure_control_mode(None, None)  # type: ignore[call-arg]

    def _start_loaded_game(self) -> Dict[str, Any]:
        self._ensure_lineups_are_valid()

        if not self.home_team or not self.away_team:
            raise GameSessionError("Teams could not be loaded from the data files.")

        self.game_state = GameState(self.home_team, self.away_team)
        reset_analytics = getattr(self, "_reset_live_analytics", None)
        if callable(reset_analytics):
            reset_analytics()
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

        control_state = None
        if hasattr(self, "get_control_state"):
            try:
                control_state = self.get_control_state()
            except Exception:
                control_state = None

        if isinstance(control_state, dict) and control_state.get("mode") == "cpu":
            user_key = control_state.get("user_team")
            cpu_key = control_state.get("cpu_team")

            def resolve_name(key: Optional[str]) -> str:
                if key == "home":
                    return getattr(self.home_team, "name", "Home")
                if key == "away":
                    return getattr(self.away_team, "name", "Away")
                return "-"

            user_team_name = control_state.get("user_team_name") or resolve_name(user_key)
            cpu_team_name = control_state.get("cpu_team_name") or resolve_name(cpu_key)
            summary = f"🎮 CPU対戦モード: あなた={user_team_name} / CPU={cpu_team_name}"
            self._log.append(summary, variant="info")
            self._notifications.publish("info", summary)
        elif isinstance(control_state, dict) and control_state.get("mode") == "auto":
            home_name = getattr(self.home_team, "name", "Home")
            away_name = getattr(self.away_team, "name", "Away")
            summary = (
                "🤖 全自動CPUモード: {away} (Away) vs {home} (Home)。"
                "進行ボタンでCPU同士の対戦を進めます。"
            ).format(away=away_name, home=home_name)
            self._log.append(summary, variant="info")
            self._notifications.publish("info", summary)

        self._log.append("=" * 60, variant="highlight")
        banner = half_inning_banner(self.game_state, self.home_team, self.away_team)
        self._log.extend_banner(banner)
        return self.build_state()

    def start_new_game(
        self,
        reload_teams: bool = False,
        *,
        control_mode: Optional[str] = None,
        user_team: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new :class:`GameState` and reset bookkeeping."""

        self._prepare_game_setup(control_mode, user_team)
        self.ensure_teams(force_reload=reload_teams)
        return self._start_loaded_game()

    def stop_game(self) -> Dict[str, Any]:
        """Exit the current game and return to the title screen."""

        self.game_state = None
        reset_analytics = getattr(self, "_reset_live_analytics", None)
        if callable(reset_analytics):
            reset_analytics()
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
        reset_analytics = getattr(self, "_reset_live_analytics", None)
        if callable(reset_analytics):
            reset_analytics()
        return self.build_state()

    def _resolve_team_key(self, team_key: str) -> Tuple[object, str]:
        """Return the requested team object and its normalized key."""

        normalized = str(team_key or "").lower()
        if normalized == "home":
            team = self.home_team
        elif normalized == "away":
            team = self.away_team
        else:
            raise GameSessionError("チームには 'home' または 'away' を指定してください。")

        if team is None:
            raise GameSessionError("指定したチームが読み込めませんでした。チームデータを再確認してください。")
        return team, normalized

    def update_starting_lineup(
        self, team_key: str, lineup_payload: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Update the pre-game starting lineup for the selected team."""

        if self.game_state:
            raise GameSessionError("試合中はスタメンを変更できません。タイトル画面に戻ってから再設定してください。")

        team, _ = self._resolve_team_key(team_key)

        if not isinstance(lineup_payload, list) or not lineup_payload:
            raise GameSessionError("スタメンデータの形式が不正です。再度やり直してください。")

        required_slots = len(getattr(team, "required_positions", [])) or 9

        normalized_entries: List[Tuple[str, str]] = []
        for entry in lineup_payload:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            position = str(entry.get("position") or "").strip().upper()
            if not name or not position:
                raise GameSessionError("スタメンには選手名とポジションの両方が必要です。")
            normalized_entries.append((name, position))

        if len(normalized_entries) != required_slots:
            raise GameSessionError(f"スタメンは{required_slots}枠すべてを埋めてください。")

        seen_names = set()
        for name, _ in normalized_entries:
            if name in seen_names:
                raise GameSessionError(f"{name} が複数の打順に設定されています。")
            seen_names.add(name)

        available_players: Dict[str, Any] = {}
        for player in list(getattr(team, "lineup", [])) + list(getattr(team, "bench", [])):
            if not player:
                continue
            if player.name not in available_players:
                available_players[player.name] = player

        new_lineup: List[Any] = []
        defensive_positions = {pos: None for pos in getattr(team, "required_positions", [])}

        for name, position in normalized_entries:
            player = available_players.get(name)
            if not player:
                raise GameSessionError(f"{team.name} のロスターに {name} が見つかりません。")
            if defensive_positions.get(position):
                raise GameSessionError(f"{position} の枠が重複しています。")
            if hasattr(player, "can_play_position") and not player.can_play_position(position):
                raise GameSessionError(f"{player.name} は {position} を守れません。")
            player.current_position = position
            defensive_positions[position] = player
            new_lineup.append(player)

        original_lineup = list(team.lineup)
        original_bench = list(team.bench)
        original_positions = dict(team.defensive_positions)
        original_batter_index = team.current_batter_index

        team.lineup = new_lineup
        team.defensive_positions = defensive_positions
        team.current_batter_index = 0

        bench_candidates = list(original_bench) + list(original_lineup)
        bench_result: List[Any] = []
        used_ids = {id(player) for player in new_lineup}
        for player in bench_candidates:
            if not player or id(player) in used_ids:
                continue
            player.current_position = None
            if player not in bench_result:
                bench_result.append(player)
        team.bench = bench_result
        if isinstance(getattr(team, "ejected_players", None), list):
            team.ejected_players.clear()

        errors = team.validate_lineup()
        if errors:
            team.lineup = original_lineup
            team.bench = original_bench
            team.defensive_positions = original_positions
            team.current_batter_index = original_batter_index
            raise GameSessionError("ラインナップが不正です: " + ", ".join(errors))

        self._notifications.publish("success", f"{team.name}のスタメンを更新しました。")
        return self.build_state()

    def set_starting_pitcher(self, team_key: str, pitcher_name: str) -> Dict[str, Any]:
        """Select the starting pitcher before the game begins."""

        if self.game_state:
            raise GameSessionError("試合中は先発投手を変更できません。タイトル画面に戻ってから再設定してください。")

        team, _ = self._resolve_team_key(team_key)
        name = str(pitcher_name or "").strip()
        if not name:
            raise GameSessionError("先発投手を選択してください。")

        candidates: List[Any] = []
        if getattr(team, "current_pitcher", None):
            candidates.append(team.current_pitcher)
        candidates.extend(getattr(team, "pitchers", []) or [])

        selected = None
        for pitcher in candidates:
            if pitcher and getattr(pitcher, "name", None) == name:
                selected = pitcher
                break

        if not selected:
            raise GameSessionError(f"{team.name} の投手 '{name}' が見つかりません。")

        if team.current_pitcher is selected:
            self._notifications.publish("info", f"{selected.name} は既に先発に設定されています。")
            return self.build_state()

        previous = getattr(team, "current_pitcher", None)
        team.current_pitcher = selected
        selected.current_position = "P"

        if previous and previous is not selected and previous not in team.pitchers:
            team.pitchers.append(previous)

        if selected not in team.pitchers:
            team.pitchers.insert(0, selected)

        if selected in getattr(team, "ejected_players", []):
            team.ejected_players.remove(selected)

        self._notifications.publish("success", f"{team.name}の先発投手を{selected.name}に設定しました。")
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

