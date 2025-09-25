"""Simulation helpers for :class:`WebGameSession`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from baseball_sim.interface.simulation import simulate_games

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

        summary = summarize_simulation_results(
            results,
            home_team=self.home_team,
            away_team=self.away_team,
            num_games=num_games,
        )

        self._simulation_state["last_run"] = summary

        self._notifications.publish(
            "success",
            f"{num_games}試合のシミュレーションが完了しました。",
        )

        return self.build_state()

