"""Simulation helpers for :class:`WebGameSession`."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from baseball_sim.interface.simulation import simulate_games
from baseball_sim.data.team_library import TeamLibraryError

from .exceptions import GameSessionError
from .simulation_summary import summarize_simulation_results


class SimulationControlsMixin:
    """Provide background simulation support for the web UI."""

    _simulation_state: Dict[str, Any]
    _home_team_source: Optional[Dict[str, Any]]
    _away_team_source: Optional[Dict[str, Any]]

    def _clear_simulation_results(self) -> None:
        self._simulation_state["last_run"] = None
        self._simulation_state["log"] = []
        self._simulation_state["running"] = False
        self._simulation_state["league"] = {
            "teams": [],
            "games_per_card": None,
            "cards_per_opponent": None,
        }

    def get_simulation_state(self) -> Dict[str, Any]:
        state = {
            "enabled": bool(self.home_team and self.away_team),
            "running": bool(self._simulation_state.get("running", False)),
            "default_games": int(self._simulation_state.get("default_games", 20) or 20),
            "limits": self._simulation_state.get("limits", {"min_games": 1, "max_games": 200}),
            "last_run": self._simulation_state.get("last_run"),
            "log": list(self._simulation_state.get("log", []))[-20:],
        }
        league_state = self._simulation_state.get("league") or {}
        state["league"] = {
            "teams": list(league_state.get("teams", [])),
            "games_per_card": league_state.get("games_per_card"),
            "cards_per_opponent": league_state.get("cards_per_opponent"),
        }
        return state

    def _prepare_league_request(
        self, league_options: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(league_options, Mapping):
            raise GameSessionError("リーグ設定の形式が不正です。")

        raw_teams = league_options.get("teams")
        if not isinstance(raw_teams, (list, tuple)) or len(raw_teams) < 2:
            raise GameSessionError("リーグには2チーム以上の参加が必要です。")

        team_datas: List[Dict[str, Any]] = []
        team_ids: List[str] = []
        for entry in raw_teams:
            team_payload: Optional[Mapping[str, Any]] = None
            team_id: Optional[str] = None
            if isinstance(entry, Mapping):
                maybe_id = entry.get("id") or entry.get("team_id")
                if isinstance(maybe_id, str):
                    team_id = maybe_id.strip()
                raw_payload = entry.get("data") or entry.get("team")
                if isinstance(raw_payload, Mapping):
                    team_payload = raw_payload
            elif isinstance(entry, str):
                team_id = entry.strip()
            else:
                raise GameSessionError("リーグチームの指定が不正です。")

            if team_payload is None:
                if not team_id:
                    raise GameSessionError("リーグチームのIDを指定してください。")
                try:
                    team_payload = self._team_library.load_team(team_id)
                except TeamLibraryError as exc:  # pragma: no cover - pass through
                    raise GameSessionError(str(exc)) from exc
            team_datas.append(dict(team_payload))
            team_ids.append(team_id or str(len(team_datas)))

        games_per_card = int(league_options.get("games_per_card", 0) or 0)
        cards_per_opponent = int(league_options.get("cards_per_opponent", 0) or 0)
        if games_per_card <= 0 or cards_per_opponent <= 0:
            raise GameSessionError("カード設定は1以上の数値で指定してください。")

        role_assignment_raw = league_options.get("role_assignment")
        role_assignment: Dict[str, int] = {}
        if isinstance(role_assignment_raw, Mapping):
            for role, index in role_assignment_raw.items():
                try:
                    normalized_index = int(index)
                except (TypeError, ValueError):
                    continue
                if 0 <= normalized_index < len(team_datas):
                    role_assignment[str(role)] = normalized_index

        league_request: Dict[str, Any] = {
            "teams": team_datas,
            "games_per_card": games_per_card,
            "cards_per_opponent": cards_per_opponent,
        }
        if role_assignment:
            league_request["role_assignment"] = role_assignment

        league_context: Dict[str, Any] = {
            "teams": team_ids,
            "games_per_card": games_per_card,
            "cards_per_opponent": cards_per_opponent,
        }
        if role_assignment:
            league_context["role_assignment"] = role_assignment

        return league_request, league_context

    def run_simulation(
        self, num_games: Optional[int] = None, *, league_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        league_request: Optional[Dict[str, Any]] = None
        league_context: Optional[Dict[str, Any]] = None

        if league_options:
            league_request, league_context = self._prepare_league_request(league_options)
        else:
            if num_games is None:
                raise GameSessionError("シミュレーションする試合数を指定してください。")

        limits = self._simulation_state.get("limits", {"min_games": 1, "max_games": 200})
        min_games = int(limits.get("min_games", 1) or 1)
        max_games = int(limits.get("max_games", 200) or 200)

        if league_request is None:
            if num_games < min_games:
                raise GameSessionError(f"シミュレーション試合数は{min_games}以上で指定してください。")
            if num_games > max_games:
                raise GameSessionError(f"シミュレーション試合数は最大{max_games}試合までです。")

        self.ensure_teams()
        if league_request is None:
            if not self.home_team or not self.away_team:
                raise GameSessionError("チームが読み込まれていません。チームを選択してください。")

            if not self._home_team_source or not self._away_team_source:
                raise GameSessionError("チームデータを読み込めませんでした。チーム選択を確認してください。")

        self._simulation_state["running"] = True

        if league_request is not None:
            progress_messages: List[str] = ["Starting league simulation..."]
        else:
            progress_messages = [f"Simulating {num_games} games..."]

        def handle_message(message: str) -> None:
            if not message:
                return
            progress_messages.append(str(message))

        try:
            if league_request is not None:
                results = simulate_games(
                    num_games=None,
                    output_file=None,
                    progress_callback=None,
                    message_callback=handle_message,
                    league_options=league_request,
                    save_to_file=False,
                )
            else:
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
        if league_context is not None:
            self._simulation_state["default_games"] = int(league_context.get("games_per_card", 0) or 0)
            self._simulation_state["league"] = league_context
        else:
            self._simulation_state["default_games"] = num_games
            self._simulation_state["league"] = {
                "teams": [self._home_team_id, self._away_team_id],
                "games_per_card": num_games,
                "cards_per_opponent": 1,
            }
        self._simulation_state["log"] = progress_messages[-20:]

        summary = summarize_simulation_results(
            results,
            home_team=self.home_team,
            away_team=self.away_team,
            num_games=num_games,
            league=league_context,
        )

        self._simulation_state["last_run"] = summary

        self._notifications.publish(
            "success",
            f"{num_games}試合のシミュレーションが完了しました。",
        )

        return self.build_state()

