"""Utilities for managing per-team JSON definitions for the web UI."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from baseball_sim.config import path_manager
from baseball_sim.config.paths import FileUtils


class TeamLibraryError(RuntimeError):
    """Raised when team library operations fail."""


_ID_SANITIZE_PATTERN = re.compile(r"[\\/:*?\"<>|]+")


def _sanitize_team_id(candidate: str, fallback: str = "team") -> str:
    """Create a filesystem-safe identifier from a team name."""

    if candidate is None:
        candidate = ""
    normalized = unicodedata.normalize("NFKC", str(candidate)).strip()
    sanitized = _ID_SANITIZE_PATTERN.sub("_", normalized)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = fallback
    # Limit length to avoid extremely long filenames
    return sanitized[:60]


@dataclass(frozen=True)
class TeamRecord:
    """Metadata about a stored team definition."""

    team_id: str
    name: str
    filename: str
    path: Path
    updated_at: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.team_id,
            "name": self.name,
            "filename": self.filename,
            "updated_at": self.updated_at,
        }


class TeamLibrary:
    """Manage single-team JSON files used by the browser UI."""

    def __init__(self) -> None:
        self._directory = Path(path_manager.get_team_library_path())
        path_manager.ensure_directory_exists(str(self._directory))
        self._selection_path = Path(path_manager.get_team_selection_path())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def ensure_initialized(self) -> None:
        """Create default team files and selection when missing."""

        path_manager.ensure_directory_exists(str(self._directory))

        has_teams = any(self._directory.glob("*.json"))
        if not has_teams:
            combined_path = path_manager.get_teams_data_path()
            combined = FileUtils.safe_json_load(combined_path, default=None)
            if not isinstance(combined, dict):
                return

            defaults: List[tuple[str, Dict[str, object]]] = []
            for key, fallback in (("home_team", "home-team"), ("away_team", "away-team")):
                team_data = combined.get(key)
                if not isinstance(team_data, dict):
                    continue
                team_id = self._generate_unique_id(
                    _sanitize_team_id(team_data.get("name", ""), fallback=fallback)
                )
                self._write_team_file(team_id, team_data, overwrite=False)
                defaults.append((key, team_id))

            if defaults:
                selection = {
                    "home": defaults[0][1],
                    "away": defaults[1][1] if len(defaults) > 1 else defaults[0][1],
                }
                FileUtils.safe_json_save(selection, str(self._selection_path))
        else:
            if not self._selection_path.exists():
                self.ensure_selection_valid()
            else:
                self.ensure_selection_valid()

    def list_teams(self) -> List[TeamRecord]:
        """Return metadata for all stored team definitions."""

        records: List[TeamRecord] = []
        for file_path in sorted(self._directory.glob("*.json")):
            data = FileUtils.safe_json_load(str(file_path), default=None)
            name = ""
            if isinstance(data, dict):
                name_value = data.get("name")
                if isinstance(name_value, str):
                    name = name_value.strip()
            if not name:
                name = file_path.stem
            try:
                updated_at = file_path.stat().st_mtime
            except OSError:
                updated_at = 0.0
            records.append(
                TeamRecord(
                    team_id=file_path.stem,
                    name=name,
                    filename=file_path.name,
                    path=file_path,
                    updated_at=updated_at,
                )
            )
        return records

    def describe(self) -> Dict[str, object]:
        """Return a serialisable description of the library state."""

        teams = self.list_teams()
        selection = self.get_selection()
        team_ids = {team.team_id for team in teams}
        home_missing = selection["home"] not in team_ids if selection["home"] else True
        away_missing = selection["away"] not in team_ids if selection["away"] else True

        ready = bool(teams) and not home_missing and not away_missing
        hint = ""
        if not teams:
            hint = "チームデータが見つかりません。チームを作成してください。"
        elif not ready:
            missing_labels = []
            if home_missing:
                missing_labels.append("ホームチーム")
            if away_missing:
                missing_labels.append("アウェイチーム")
            hint = " / ".join(missing_labels) + "の設定が必要です。"
        else:
            hint = "チームを選択して試合画面に進めます。"

        return {
            "teams": [record.to_dict() for record in teams],
            "selection": selection,
            "ready": ready,
            "missing": {"home": home_missing, "away": away_missing},
            "hint": hint,
        }

    def get_selection(self) -> Dict[str, Optional[str]]:
        """Return the currently selected team identifiers."""

        data = FileUtils.safe_json_load(str(self._selection_path), default={})
        home = data.get("home")
        away = data.get("away")
        result = {
            "home": str(home) if isinstance(home, str) and home else None,
            "away": str(away) if isinstance(away, str) and away else None,
        }
        return result

    def set_selection(self, home_id: str, away_id: str) -> Dict[str, str]:
        """Persist the identifiers for the active teams."""

        team_ids = {record.team_id for record in self.list_teams()}
        if home_id not in team_ids:
            raise TeamLibraryError(f"ホームチーム '{home_id}' が見つかりません。")
        if away_id not in team_ids:
            raise TeamLibraryError(f"アウェイチーム '{away_id}' が見つかりません。")

        selection = {"home": home_id, "away": away_id}
        if not FileUtils.safe_json_save(selection, str(self._selection_path)):
            raise TeamLibraryError("チーム選択の保存に失敗しました。")
        return selection

    def load_team(self, team_id: str) -> Dict[str, object]:
        """Load a single team definition by identifier."""

        file_path = self._directory / f"{team_id}.json"
        if not file_path.is_file():
            raise TeamLibraryError(f"チーム '{team_id}' のファイルが見つかりません。")

        data = FileUtils.safe_json_load(str(file_path), default=None)
        if not isinstance(data, dict):
            raise TeamLibraryError(f"チーム '{team_id}' のデータを読み込めませんでした。")
        return data

    def save_team(self, team_id: Optional[str], team_data: Dict[str, object]) -> str:
        """Create or update a team definition and return its identifier."""

        normalised = self._normalise_team_payload(team_data)

        if team_id:
            candidate_id = _sanitize_team_id(team_id)
            if not candidate_id:
                candidate_id = _sanitize_team_id(normalised.get("name", ""))
        else:
            candidate_id = _sanitize_team_id(normalised.get("name", ""))

        if not candidate_id:
            candidate_id = "team"

        if not team_id:
            candidate_id = self._generate_unique_id(candidate_id)

        self._write_team_file(candidate_id, normalised, overwrite=True)
        return candidate_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_unique_id(self, base_id: str) -> str:
        candidate = base_id or "team"
        counter = 1
        while (self._directory / f"{candidate}.json").exists():
            counter += 1
            candidate = f"{base_id}-{counter}"
        return candidate

    def _write_team_file(self, team_id: str, payload: Dict[str, object], *, overwrite: bool) -> None:
        target_path = self._directory / f"{team_id}.json"
        if not overwrite and target_path.exists():
            return
        if not FileUtils.safe_json_save(payload, str(target_path)):
            raise TeamLibraryError(f"チーム '{team_id}' の保存に失敗しました。")

    def ensure_selection_valid(self) -> Dict[str, str]:
        teams = self.list_teams()
        if not teams:
            selection = {"home": None, "away": None}
            FileUtils.safe_json_save(selection, str(self._selection_path))
            return selection

        ids = [record.team_id for record in teams]
        current = self.get_selection()
        home = current.get("home")
        away = current.get("away")

        if home not in ids:
            home = ids[0]
        if away not in ids:
            away = ids[1] if len(ids) > 1 else ids[0]

        selection = {"home": home, "away": away}
        FileUtils.safe_json_save(selection, str(self._selection_path))
        return selection

    def _normalise_team_payload(self, raw: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise TeamLibraryError("チームデータの形式が不正です。")

        name = str(raw.get("name", "")).strip()
        if not name:
            raise TeamLibraryError("チーム名を入力してください。")

        lineup_raw = raw.get("lineup")
        if not isinstance(lineup_raw, list):
            raise TeamLibraryError("ラインナップは配列で指定してください。")

        lineup: List[Dict[str, object]] = []
        for entry in lineup_raw:
            if not isinstance(entry, dict):
                raise TeamLibraryError("ラインナップの各要素はオブジェクトである必要があります。")
            player_name = str(entry.get("name", "")).strip()
            position = str(entry.get("position", "")).strip()
            if not player_name:
                raise TeamLibraryError("ラインナップの選手名が未入力です。")
            if not position:
                raise TeamLibraryError(f"{player_name} のポジションを入力してください。")
            normalised_entry = dict(entry)
            normalised_entry["name"] = player_name
            normalised_entry["position"] = position
            lineup.append(normalised_entry)

        pitchers_raw = raw.get("pitchers")
        if not isinstance(pitchers_raw, list):
            raise TeamLibraryError("投手リストは配列で指定してください。")
        pitchers = [str(name).strip() for name in pitchers_raw if str(name).strip()]
        if not pitchers:
            raise TeamLibraryError("最低1人の投手を登録してください。")

        bench_raw = raw.get("bench")
        if not isinstance(bench_raw, list):
            raise TeamLibraryError("ベンチリストは配列で指定してください。")
        bench = [str(name).strip() for name in bench_raw if str(name).strip()]

        payload = dict(raw)
        payload["name"] = name
        payload["lineup"] = lineup
        payload["pitchers"] = pitchers
        payload["bench"] = bench
        return payload


__all__ = ["TeamLibrary", "TeamLibraryError"]
