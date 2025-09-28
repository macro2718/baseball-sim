"""Project path helpers used throughout the application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .constants import FilePaths


class FileUtils:
    """Resilient JSON helpers used by multiple modules."""

    @staticmethod
    def safe_json_load(filepath: Path | str, default: Any = None) -> Any:
        """Load JSON returning ``default`` when the file is missing or invalid."""

        try:
            path = Path(filepath)
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def safe_json_save(data: Any, filepath: Path | str) -> bool:
        """Persist JSON data, returning ``True`` on success."""

        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            return True
        except (OSError, TypeError) as exc:
            print(f"Error saving JSON to {path}: {exc}")
            return False


class ProjectPaths:
    """Locate important directories and files for the project."""

    def __init__(self) -> None:
        self._package_dir = Path(__file__).resolve().parent.parent
        self._project_root = self._package_dir.parent

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def player_data_dir(self) -> Path:
        return self._project_root / "player_data"

    @property
    def prediction_models_dir(self) -> Path:
        return self._project_root / "prediction_models"

    @property
    def data_dir(self) -> Path:
        return self.player_data_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.prediction_models_dir / "models"

    @property
    def players_data_file(self) -> Path:
        return self.data_dir / FilePaths.PLAYERS_JSON

    @property
    def teams_data_file(self) -> Path:
        return self.data_dir / FilePaths.TEAMS_JSON

    @property
    def team_library_dir(self) -> Path:
        return self.player_data_dir / FilePaths.TEAM_LIBRARY_DIR

    @property
    def team_selection_file(self) -> Path:
        return self.data_dir / FilePaths.TEAM_SELECTION_JSON

    @property
    def batting_model_file(self) -> Path:
        return self.models_dir / FilePaths.BATTING_MODEL

    @property
    def nn_model_file(self) -> Path:
        return self.models_dir / FilePaths.NN_MODEL

    def get_data_path(self, filename: str | None = None) -> Path:
        return self.data_dir / filename if filename else self.data_dir

    def get_models_path(self, filename: str | None = None) -> Path:
        return self.models_dir / filename if filename else self.models_dir

    def get_players_data_path(self) -> Path:
        return self.players_data_file

    def get_teams_data_path(self) -> Path:
        return self.teams_data_file

    def get_team_library_path(self, filename: str | None = None) -> Path:
        base = self.team_library_dir
        return base / filename if filename else base

    def get_team_selection_path(self) -> Path:
        return self.team_selection_file

    def get_batting_model_path(self) -> Path:
        return self.batting_model_file

    def get_nn_model_path(self) -> Path:
        return self.nn_model_file

    def ensure_directory_exists(self, path: Path | str) -> bool:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def file_exists(self, path: Path | str) -> bool:
        return Path(path).is_file()

    def ensure_project_root_on_sys_path(self) -> None:
        """Guarantee that the project root is importable."""

        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)


_project_paths = ProjectPaths()


def setup_project_environment() -> None:
    """Backwards compatible environment bootstrapper."""

    _project_paths.ensure_project_root_on_sys_path()


def get_project_paths() -> ProjectPaths:
    return _project_paths


setup_project_environment()
