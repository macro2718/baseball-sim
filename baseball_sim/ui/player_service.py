"""Utilities for managing player data for the web API.

This module extracts the heavy helper logic from ``routes.py`` so that the
route declarations remain lightweight.  The service is intentionally focused on
filesystem interactions (loading/saving player catalogues and updating team
references) and keeps the HTTP specific concerns inside the Flask handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import uuid

from baseball_sim.config import FileUtils, get_project_paths


PATHS = get_project_paths()


@dataclass
class PlayerDataset:
    """Container for the players JSON document."""

    data: Dict[str, List[dict]]
    path: Path
    mutated: bool = False


@dataclass
class PlayerSaveResult:
    """Outcome of saving a player."""

    player_id: str
    player_name: str
    role: str
    previous_name: Optional[str]


class PlayerLibraryError(RuntimeError):
    """Raised when an operation on the player library cannot be completed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PlayerLibraryService:
    """Handles persistence and reference management for players."""

    def __init__(self, *, file_utils: type[FileUtils] = FileUtils):
        self._file_utils = file_utils

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_dataset(self) -> PlayerDataset:
        """Load ``players.json`` and ensure each entry has a stable id.

        Also migrates structure to support optional folder assignments:
        - Adds a top-level "folders" list if missing
        - Ensures each player has an optional "folders" list (if present and valid)
        """

        players_path = Path(PATHS.get_players_data_path())
        raw = self._file_utils.safe_json_load(
            players_path, default={"batters": [], "pitchers": [], "folders": []}
        )
        if not isinstance(raw, dict):
            raw = {"batters": [], "pitchers": [], "folders": []}

        raw.setdefault("batters", [])
        raw.setdefault("pitchers", [])
        raw.setdefault("folders", [])

        mutated = False
        for role_key in ("batters", "pitchers"):
            items = raw.get(role_key, []) or []
            for record in items:
                if isinstance(record, dict) and not record.get("id"):
                    record["id"] = str(uuid.uuid4())
                    mutated = True
                # Normalise folders to a list of non-empty strings
                if isinstance(record, dict):
                    folders = record.get("folders")
                    if folders is None:
                        continue
                    if isinstance(folders, list):
                        cleaned = []
                        for f in folders:
                            if isinstance(f, str):
                                name = f.strip()
                                if name:
                                    cleaned.append(name)
                        record["folders"] = cleaned
                    else:
                        # Drop invalid folder format
                        record.pop("folders", None)
                        mutated = True

        # Normalise top-level folders list
        if not isinstance(raw.get("folders"), list):
            raw["folders"] = []
            mutated = True
        else:
            cleaned = []
            seen = set()
            for f in raw["folders"]:
                if not isinstance(f, str):
                    continue
                name = f.strip()
                if not name or name in seen:
                    continue
                cleaned.append(name)
                seen.add(name)
            if cleaned != raw["folders"]:
                raw["folders"] = cleaned
                mutated = True

        return PlayerDataset(data=raw, path=players_path, mutated=mutated)

    def save_dataset(self, dataset: PlayerDataset) -> bool:
        return self._file_utils.safe_json_save(dataset.data, dataset.path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def find_player_by_id(
        self, dataset: PlayerDataset, player_id: str
    ) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
        if not player_id:
            return None, None, None
        for role_key in ("batters", "pitchers"):
            items: List[dict] = dataset.data.get(role_key, []) or []
            for idx, record in enumerate(items):
                if isinstance(record, dict) and record.get("id") == player_id:
                    return record, role_key, idx
        return None, None, None

    def find_player_by_name(
        self, dataset: PlayerDataset, name: str
    ) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
        if not name:
            return None, None, None
        for role_key in ("batters", "pitchers"):
            items: List[dict] = dataset.data.get(role_key, []) or []
            for idx, record in enumerate(items):
                if isinstance(record, dict) and record.get("name") == name:
                    return record, role_key, idx
        return None, None, None

    # ------------------------------------------------------------------
    # Catalogue helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_positions(raw_positions: object) -> List[str]:
        positions: List[str] = []
        if isinstance(raw_positions, list):
            for pos in raw_positions:
                if not isinstance(pos, str):
                    continue
                token = pos.strip().upper()
                if token and token not in positions:
                    positions.append(token)
        return positions

    def build_catalogue(self, dataset: PlayerDataset) -> Dict[str, List[dict]]:
        batters: List[dict] = []
        for record in dataset.data.get("batters", []) or []:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            pid = record.get("id")
            if not name or not pid:
                continue
            batters.append(
                {
                    "id": str(pid),
                    "name": str(name),
                    "bats": (record.get("bats") or "").strip().upper() or None,
                    "eligible_positions": self._normalise_positions(
                        record.get("eligible_positions")
                    ),
                    "k_pct": self._safe_float(record.get("k_pct")),
                    "bb_pct": self._safe_float(record.get("bb_pct")),
                    "hard_pct": self._safe_float(record.get("hard_pct")),
                    "gb_pct": self._safe_float(record.get("gb_pct")),
                    "speed": self._safe_float(record.get("speed")),
                    "fielding_skill": self._safe_float(record.get("fielding_skill")),
                    "folders": [
                        f for f in (record.get("folders") or []) if isinstance(f, str) and f.strip()
                    ],
                }
            )

        pitchers: List[dict] = []
        for record in dataset.data.get("pitchers", []) or []:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            pid = record.get("id")
            if not name or not pid:
                continue
            pitchers.append(
                {
                    "id": str(pid),
                    "name": str(name),
                    "pitcher_type": (record.get("pitcher_type") or "")
                    .strip()
                    .upper()
                    or None,
                    "throws": (record.get("throws") or "").strip().upper() or None,
                    "k_pct": self._safe_float(record.get("k_pct")),
                    "bb_pct": self._safe_float(record.get("bb_pct")),
                    "hard_pct": self._safe_float(record.get("hard_pct")),
                    "gb_pct": self._safe_float(record.get("gb_pct")),
                    "stamina": self._safe_float(record.get("stamina")),
                    "folders": [
                        f for f in (record.get("folders") or []) if isinstance(f, str) and f.strip()
                    ],
                }
            )

        return {"batters": batters, "pitchers": pitchers}

    def build_list(self, dataset: PlayerDataset, role: str, *, folder: str | None = None) -> Dict[str, List[dict]]:
        def pack(items: Iterable[dict]) -> List[dict]:
            output: List[dict] = []
            for record in items:
                if not isinstance(record, dict):
                    continue
                name = record.get("name")
                pid = record.get("id")
                if not name or not pid:
                    continue
                if folder and isinstance(record.get("folders"), list):
                    # record folders are strings
                    folders = [
                        f for f in record.get("folders") if isinstance(f, str) and f.strip()
                    ]
                    if folder not in folders:
                        continue
                output.append({"id": str(pid), "name": str(name)})
            return output

        key = "pitchers" if role == "pitcher" else "batters"
        return {"players": pack(list(dataset.data.get(key, [])))}

    # ------------------------------------------------------------------
    # Folder helpers
    # ------------------------------------------------------------------
    def list_folders(self, dataset: PlayerDataset) -> List[str]:
        folders = dataset.data.get("folders")
        if not isinstance(folders, list):
            return []
        return [str(f).strip() for f in folders if isinstance(f, str) and str(f).strip()]

    def add_folder(self, dataset: PlayerDataset, name: str) -> List[str]:
        cleaned = str(name or "").strip()
        if not cleaned:
            raise PlayerLibraryError("タグ名を入力してください。")
        folders = dataset.data.setdefault("folders", [])
        if not isinstance(folders, list):
            folders = []
        if cleaned not in folders:
            folders.append(cleaned)
            dataset.data["folders"] = folders
            dataset.mutated = True
        return self.list_folders(dataset)

    def delete_folder(self, dataset: PlayerDataset, name: str) -> List[str]:
        target = str(name or "").strip()
        if not target:
            raise PlayerLibraryError("削除するタグ名を指定してください。")
        folders = dataset.data.get("folders")
        if not isinstance(folders, list) or target not in folders:
            raise PlayerLibraryError("指定したタグは存在しません。")
        # Remove from master folder list
        dataset.data["folders"] = [f for f in folders if f != target]
        # Remove from all players
        for role_key in ("batters", "pitchers"):
            for record in dataset.data.get(role_key, []) or []:
                if isinstance(record, dict) and isinstance(record.get("folders"), list):
                    record["folders"] = [f for f in record["folders"] if isinstance(f, str) and f.strip() and f.strip() != target]
        dataset.mutated = True
        return self.list_folders(dataset)

    def rename_folder(self, dataset: PlayerDataset, old_name: str, new_name: str) -> List[str]:
        old_clean = str(old_name or "").strip()
        new_clean = str(new_name or "").strip()
        if not old_clean or not new_clean:
            raise PlayerLibraryError("タグ名を正しく入力してください。")
        if old_clean == new_clean:
            return self.list_folders(dataset)
        folders = dataset.data.get("folders")
        if not isinstance(folders, list) or old_clean not in folders:
            raise PlayerLibraryError("指定したタグは存在しません。")
        if new_clean in folders:
            raise PlayerLibraryError("同名のタグが既に存在します。")
        # Replace in master folder list
        dataset.data["folders"] = [new_clean if f == old_clean else f for f in folders]
        # Replace in all players
        for role_key in ("batters", "pitchers"):
            for record in dataset.data.get(role_key, []) or []:
                if isinstance(record, dict) and isinstance(record.get("folders"), list):
                    new_list: List[str] = []
                    for f in record["folders"]:
                        if not isinstance(f, str):
                            continue
                        name = f.strip()
                        if not name:
                            continue
                        if name == old_clean:
                            if new_clean not in new_list:
                                new_list.append(new_clean)
                        else:
                            if name not in new_list:
                                new_list.append(name)
                    record["folders"] = new_list
        dataset.mutated = True
        return self.list_folders(dataset)

    # ------------------------------------------------------------------
    # Reference helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_team_name_slots(
        team_obj: dict,
    ) -> Iterator[Tuple[object, object, Callable[[], Optional[str]], Callable[[str], None]]]:
        if not isinstance(team_obj, dict):
            return

        def list_setter(container: List[object], index: int) -> Callable[[str], None]:
            return lambda value: container.__setitem__(index, value)

        def list_getter(container: List[object], index: int) -> Callable[[], Optional[str]]:
            def getter() -> Optional[str]:
                value = container[index]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    name = value.get("name")
                    return name if isinstance(name, str) else None
                return None

            return getter

        for key in ("lineup", "bench", "pitchers", "rotation"):
            entries = team_obj.get(key)
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                if isinstance(entry, str):
                    yield entries, index, list_getter(entries, index), list_setter(entries, index)
                elif isinstance(entry, dict):
                    def dict_getter(obj: dict = entry) -> Optional[str]:
                        name = obj.get("name")
                        return name if isinstance(name, str) else None

                    def dict_setter(value: str, obj: dict = entry) -> None:
                        obj["name"] = value

                    yield entry, "name", dict_getter, dict_setter

    def _team_mentions_player(self, team_obj: dict, target_name: str) -> bool:
        if not isinstance(team_obj, dict) or not target_name:
            return False
        for _, _, getter, _ in self._iter_team_name_slots(team_obj):
            current_name = getter()
            if isinstance(current_name, str) and current_name == target_name:
                return True
        return False

    def find_referencing_teams(self, target_name: str) -> List[str]:
        refs: List[str] = []
        if not target_name:
            return refs

        try:
            teams_path = Path(PATHS.get_teams_data_path())
            teams_data = self._file_utils.safe_json_load(teams_path, default=None)
            if isinstance(teams_data, dict):
                for key in ("home_team", "away_team"):
                    team_obj = teams_data.get(key)
                    if isinstance(team_obj, dict) and self._team_mentions_player(
                        team_obj, target_name
                    ):
                        team_name = team_obj.get("name") or key
                        refs.append(str(team_name))
        except Exception:
            pass

        try:
            teams_dir = Path(PATHS.get_team_library_path())
            if teams_dir.is_dir():
                for file_path in teams_dir.glob("*.json"):
                    team_obj = self._file_utils.safe_json_load(file_path, default=None)
                    if isinstance(team_obj, dict) and self._team_mentions_player(
                        team_obj, target_name
                    ):
                        team_name = team_obj.get("name") or file_path.stem
                        refs.append(str(team_name))
        except Exception:
            pass

        return refs

    def _rename_player_in_team(self, team_obj: dict, old_name: str, new_name: str) -> bool:
        if not isinstance(team_obj, dict):
            return False
        if not old_name or not new_name or old_name == new_name:
            return False

        changed = False

        for container, key, getter, setter in self._iter_team_name_slots(team_obj):
            current_name = getter()
            if isinstance(current_name, str) and current_name == old_name:
                setter(new_name)
                changed = True
        return changed

    def rename_player_references(self, old_name: str, new_name: str) -> None:
        if not old_name or not new_name or old_name == new_name:
            return

        try:
            teams_path = Path(PATHS.get_teams_data_path())
            teams_data = self._file_utils.safe_json_load(teams_path, default=None)
        except Exception:
            teams_data = None

        if isinstance(teams_data, dict):
            updated = False
            for key in ("home_team", "away_team"):
                team_obj = teams_data.get(key)
                if isinstance(team_obj, dict) and self._rename_player_in_team(
                    team_obj, old_name, new_name
                ):
                    updated = True
            if updated:
                self._file_utils.safe_json_save(teams_data, teams_path)

        teams_dir = Path(PATHS.get_team_library_path())
        if teams_dir.is_dir():
            for file_path in teams_dir.glob("*.json"):
                try:
                    team_obj = self._file_utils.safe_json_load(file_path, default=None)
                except Exception:
                    continue
                if isinstance(team_obj, dict) and self._rename_player_in_team(
                    team_obj, old_name, new_name
                ):
                    self._file_utils.safe_json_save(team_obj, file_path)

    # ------------------------------------------------------------------
    # Business actions
    # ------------------------------------------------------------------
    @staticmethod
    def infer_role(player: dict, fallback: str = "batter") -> str:
        if not isinstance(player, dict):
            return fallback
        if "stamina" in player or "pitcher_type" in player:
            return "pitcher"
        return fallback

    def save_player(
        self,
        dataset: PlayerDataset,
        player: dict,
        role: str,
        *,
        player_id: str = "",
        original_name: str = "",
    ) -> PlayerSaveResult:
        if not isinstance(player, dict) or not player.get("name"):
            raise PlayerLibraryError("選手データの形式が不正です。")

        if role not in ("batter", "pitcher"):
            role = self.infer_role(player)

        existing_player = None
        existing_role_key = None
        existing_index: Optional[int] = None
        previous_name: Optional[str] = None

        if player_id:
            existing_player, existing_role_key, existing_index = self.find_player_by_id(
                dataset, player_id
            )
        elif original_name:
            existing_player, existing_role_key, existing_index = self.find_player_by_name(
                dataset, original_name
            )

        if isinstance(existing_player, dict):
            previous_name = existing_player.get("name")

        # Ensure stable id
        if existing_player and existing_player.get("id"):
            player["id"] = existing_player["id"]
        elif player_id:
            player["id"] = player_id
        else:
            player.setdefault("id", str(uuid.uuid4()))

        # Normalise folders field if provided
        if isinstance(player.get("folders"), list):
            cleaned: List[str] = []
            for f in player.get("folders"):
                if isinstance(f, str):
                    name = f.strip()
                    if name:
                        cleaned.append(name)
            player["folders"] = cleaned
            # merge into global folder list
            global_folders = dataset.data.setdefault("folders", [])
            if not isinstance(global_folders, list):
                global_folders = []
            added = False
            for f in cleaned:
                if f not in global_folders:
                    global_folders.append(f)
                    added = True
            if added:
                dataset.data["folders"] = global_folders

        target_key = "pitchers" if role == "pitcher" else "batters"

        if (
            existing_player is not None
            and existing_role_key is not None
            and existing_index is not None
        ):
            del dataset.data[existing_role_key][existing_index]
            dataset.data[target_key].append(player)
        else:
            dataset.data[target_key].append(player)

        dataset.mutated = True

        return PlayerSaveResult(
            player_id=player["id"],
            player_name=player["name"],
            role=role,
            previous_name=previous_name,
        )

    def get_player_detail(
        self, dataset: PlayerDataset, *, player_id: str = "", name: str = ""
    ) -> Dict[str, object]:
        if player_id:
            player, role_key, _ = self.find_player_by_id(dataset, player_id)
            if player and role_key:
                referenced_by = self.find_referencing_teams(player.get("name"))
                return {
                    "player": player,
                    "role": "pitcher" if role_key == "pitchers" else "batter",
                    "referenced_by": referenced_by,
                    "has_references": bool(referenced_by),
                }
            raise PlayerLibraryError("指定された選手IDは見つかりません。")

        if not name:
            raise PlayerLibraryError("選手IDまたは選手名を指定してください。")

        player, role_key, _ = self.find_player_by_name(dataset, name)
        if player and role_key:
            referenced_by = self.find_referencing_teams(player.get("name"))
            return {
                "player": player,
                "role": "pitcher" if role_key == "pitchers" else "batter",
                "referenced_by": referenced_by,
                "has_references": bool(referenced_by),
            }

        raise PlayerLibraryError("指定された選手は見つかりません。")

    def delete_player(
        self,
        dataset: PlayerDataset,
        *,
        player_id: str = "",
        name: str = "",
        role: str = "",
    ) -> str:
        player_name_for_check: Optional[str] = None

        if player_id:
            player, _, _ = self.find_player_by_id(dataset, player_id)
            if player and isinstance(player, dict):
                player_name_for_check = player.get("name") or name or None
        if not player_name_for_check:
            player_name_for_check = name or None

        referencing = self.find_referencing_teams(player_name_for_check or "")
        if referencing:
            team_list = ", ".join(referencing[:5])
            if len(referencing) > 5:
                team_list += " 他" + str(len(referencing) - 5) + "件"
            raise PlayerLibraryError(
                "選手 '{name}' はチーム({teams})に含まれているため削除できません。"
                "先に該当チームから外してください。".format(
                    name=player_name_for_check, teams=team_list
                )
            )

        removed_any = False
        if player_id:
            player, role_key, idx = self.find_player_by_id(dataset, player_id)
            if player is not None and role_key is not None and idx is not None:
                del dataset.data[role_key][idx]
                removed_any = True
        else:
            if not name:
                raise PlayerLibraryError("削除する選手IDまたは選手名を指定してください。")

            def remove_by_name(items: List[dict], target_name: str) -> Tuple[bool, List[dict]]:
                removed = False
                output: List[dict] = []
                for record in items:
                    if isinstance(record, dict) and record.get("name") == target_name:
                        removed = True
                    else:
                        output.append(record)
                return removed, output

            if role in ("batter", "pitcher"):
                key = "pitchers" if role == "pitcher" else "batters"
                removed_any, dataset.data[key] = remove_by_name(
                    list(dataset.data.get(key, [])), name
                )
            else:
                removed_bat, new_bat = remove_by_name(
                    list(dataset.data.get("batters", [])), name
                )
                removed_pit, new_pit = remove_by_name(
                    list(dataset.data.get("pitchers", [])), name
                )
                dataset.data["batters"] = new_bat
                dataset.data["pitchers"] = new_pit
                removed_any = removed_bat or removed_pit

        if not removed_any:
            raise PlayerLibraryError("指定された選手は見つかりません。")

        dataset.mutated = True
        return player_id or name
