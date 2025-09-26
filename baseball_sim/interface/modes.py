"""Helpers for selecting and executing non-web game modes."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar, Optional, Sequence

from baseball_sim.config import setup_project_environment
from baseball_sim.interface.simulation import simulate_games

setup_project_environment()


class GameModeManager:
    """Command-line utilities for executing simulator modes."""

    DEFAULT_MODE: ClassVar[str] = "simulation"
    _parsed_args: ClassVar[Optional[argparse.Namespace]] = None

    @classmethod
    def _parse_args(cls, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """Parse and cache command-line arguments for simulator execution."""

        if cls._parsed_args is None:
            parser = argparse.ArgumentParser(
                description=(
                    "Run the baseball simulator in non-web modes. "
                    "Only the automated simulation mode is currently available."
                )
            )
            parser.add_argument(
                "--mode",
                choices=[cls.DEFAULT_MODE],
                default=cls.DEFAULT_MODE,
                help="Simulator mode to execute (only 'simulation' is supported).",
            )
            parser.add_argument(
                "--games",
                type=int,
                default=10,
                help="Number of games to simulate.",
            )
            parser.add_argument(
                "--output-file",
                type=str,
                default=None,
                help=(
                    "Optional destination file for simulation results. "
                    "If omitted, a timestamped file will be created when saving is enabled."
                ),
            )
            parser.add_argument(
                "--no-save",
                action="store_true",
                help="Do not persist simulation results to disk.",
            )

            cls._parsed_args = parser.parse_args(argv)

        return cls._parsed_args

    @classmethod
    def get_game_mode_choice(cls, argv: Optional[Sequence[str]] = None) -> str:
        """Return the selected simulator mode (defaults to simulation)."""

        args = cls._parse_args(argv)
        return args.mode

    @classmethod
    def play_simulation_mode(cls) -> dict:
        """Execute the automated simulation mode and return aggregated results."""

        args = cls._parse_args()

        def progress_callback(current: int, total: int) -> None:
            print(f"Progress: {current}/{total} games", end="\r", file=sys.stdout)

        def message_callback(message: str) -> None:
            print(message, file=sys.stdout)

        results = simulate_games(
            num_games=args.games,
            output_file=args.output_file,
            progress_callback=progress_callback,
            message_callback=message_callback,
            save_to_file=not args.no_save,
        )

        print("\nSimulation complete.", file=sys.stdout)
        return results

    @classmethod
    def execute_game_mode(cls, mode: str) -> dict:
        """Execute the requested mode if supported."""

        if mode != cls.DEFAULT_MODE:
            raise ValueError(f"Unknown game mode: {mode}")

        return cls.play_simulation_mode()
