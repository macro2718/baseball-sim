"""Project path helpers used throughout the application."""

from __future__ import annotations

import sys
from pathlib import Path


class ProjectPaths:
    """Locate important directories for the project."""

    def __init__(self) -> None:
        self._package_dir = Path(__file__).resolve().parent.parent
        self._project_root = self._package_dir.parent

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def package_dir(self) -> Path:
        return self._package_dir

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
