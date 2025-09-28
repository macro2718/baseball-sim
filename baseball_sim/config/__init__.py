"""Convenience re-exports for configuration utilities.

This module exposes the most commonly used configuration helpers so that
callers can continue to rely on compact imports such as
``from baseball_sim.config import Positions`` even after the package
restructure.
"""

from .constants import (  # noqa: F401
    BASES_COUNT,
    FilePaths,
    GameResults,
    INNINGS_PER_GAME,
    MAX_EXTRA_INNINGS,
    OUTS_PER_INNING,
    PLAYERS_PER_TEAM,
    Positions,
    StatColumns,
    UIConstants,
    BuntConstants,
)
from .environment import (  # noqa: F401
    ProjectPaths,
    get_project_paths,
    setup_project_environment,
)
from .paths import FileUtils, PathManager, path_manager  # noqa: F401
from .settings import ConfigManager, config  # noqa: F401
from .league import LeagueAverages  # noqa: F401

__all__ = [
    "BASES_COUNT",
    "BuntConstants",
    "ConfigManager",
    "FilePaths",
    "FileUtils",
    "GameResults",
    "INNINGS_PER_GAME",
    "MAX_EXTRA_INNINGS",
    "OUTS_PER_INNING",
    "PLAYERS_PER_TEAM",
    "PathManager",
    "Positions",
    "ProjectPaths",
    "LeagueAverages",
    "StatColumns",
    "UIConstants",
    "config",
    "get_project_paths",
    "path_manager",
    "setup_project_environment",
]
