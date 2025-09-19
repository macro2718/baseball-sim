"""Outcome handling helpers for at-bats."""
from __future__ import annotations

from typing import Callable, Dict

from baseball_sim.gameplay.utils import RunnerEngine


class AtBatOutcomeHandler:
    """Applies at-bat results to the active :class:`GameState`."""

    def __init__(self, game_state) -> None:
        self._game_state = game_state
        self._runner_engine = RunnerEngine(game_state)
        self._handlers: Dict[str, Callable] = {
            "strikeout": self._handle_strikeout,
            "walk": self._handle_walk,
            "single": self._handle_single,
            "double": self._handle_double,
            "triple": self._handle_triple,
            "home_run": self._handle_home_run,
            "groundout": self._handle_groundout,
            "flyout": self._handle_flyout,
        }

    def apply(self, result: str, batter) -> str:
        handler = self._handlers.get(result)
        if handler is None:
            self._game_state.add_out()
            return "Out."
        return handler(batter)

    def _handle_strikeout(self, batter) -> str:
        pitcher = self._game_state.fielding_team.current_pitcher
        pitcher.pitching_stats["IP"] += 1 / 3
        batter.stats["AB"] += 1
        batter.stats["SO"] += 1
        pitcher.pitching_stats["SO"] += 1
        self._game_state.add_out()
        return "Strike out"

    def _handle_walk(self, batter) -> str:
        batter.stats["BB"] += 1
        self._game_state.fielding_team.current_pitcher.pitching_stats["BB"] += 1
        runs = self._runner_engine.apply_walk(batter)
        if runs > 0:
            self._game_state._add_runs(runs, batter)
            return f"Walk - {runs} run(s) scored"
        return "Walk"

    def _handle_single(self, batter) -> str:
        batter.stats["AB"] += 1
        batter.stats["1B"] += 1
        self._game_state.fielding_team.current_pitcher.pitching_stats["H"] += 1
        runs = self._runner_engine.apply_single(batter)
        if runs > 0:
            self._game_state._add_runs(runs, batter)
            return f"Single - {runs} run(s) scored"
        return "Single"

    def _handle_double(self, batter) -> str:
        batter.stats["AB"] += 1
        batter.stats["2B"] += 1
        self._game_state.fielding_team.current_pitcher.pitching_stats["H"] += 1
        runs = self._runner_engine.apply_double(batter)
        if runs > 0:
            self._game_state._add_runs(runs, batter)
            return f"Double! {runs} run(s) scored!"
        return "Double!"

    def _handle_triple(self, batter) -> str:
        batter.stats["AB"] += 1
        batter.stats["3B"] += 1
        self._game_state.fielding_team.current_pitcher.pitching_stats["H"] += 1
        runs = self._runner_engine.apply_triple(batter)
        if runs > 0:
            self._game_state._add_runs(runs, batter)
            return f"Triple! {runs} run(s) scored!"
        return "Triple!"

    def _handle_home_run(self, batter) -> str:
        batter.stats["AB"] += 1
        batter.stats["HR"] += 1
        pitcher = self._game_state.fielding_team.current_pitcher
        pitcher.pitching_stats["H"] += 1
        pitcher.pitching_stats["HR"] += 1
        runs = self._runner_engine.apply_home_run(batter)
        self._game_state._add_runs(runs, batter)
        return f"{runs}-run home run!" if runs > 1 else "Solo home run!"

    def _handle_groundout(self, batter) -> str:
        pitcher = self._game_state.fielding_team.current_pitcher
        pitcher.pitching_stats["IP"] += 1 / 3
        batter.stats["AB"] += 1
        runs, message = self._runner_engine.apply_groundout(batter)
        if runs > 0:
            self._game_state._add_runs(runs, batter)
        return message

    def _handle_flyout(self, batter) -> str:
        pitcher = self._game_state.fielding_team.current_pitcher
        pitcher.pitching_stats["IP"] += 1 / 3
        batter.stats["AB"] += 1
        runs = self._runner_engine.apply_flyout(batter)
        self._game_state.add_out()
        if runs > 0:
            self._game_state._add_runs(runs, batter)
            return f"Sacrifice fly! {runs} run scored!"
        return "Flyout."
