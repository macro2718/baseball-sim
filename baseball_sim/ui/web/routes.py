"""API route handlers for the baseball simulator web interface.

Web Player Editor changes:
- Players are now managed by stable unique IDs instead of names to avoid
    overwriting when multiple players share the same name.
- The players.json file will be auto-migrated to include an "id" field for
    each player the first time the Players API is used, if missing.
"""

from __future__ import annotations

from typing import Any, Dict, Callable, Tuple, Optional, List

from flask import Blueprint, jsonify, request, Response

from .session import GameSessionError, WebGameSession
from baseball_sim.config.paths import path_manager, FileUtils
import json
import uuid
import os

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')


def create_error_response(error: str, session: WebGameSession) -> Tuple[Response, int]:
    """Create a standardized error response."""
    return jsonify({"error": error, "state": session.build_state()}), 400


def parse_int_param(payload: dict, key: str, default: int = -1) -> int:
    """Safely parse integer parameter from payload."""
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def create_routes(session: WebGameSession) -> Blueprint:
    """Create and configure API routes with the given session."""
    
    @api_bp.get("/game/state")
    def get_state() -> Dict[str, Any]:
        state = session.build_state()
        return jsonify(state)

    @api_bp.post("/game/start")
    def start_game() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        reload_teams = bool(payload.get("reload", False))
        try:
            state = session.start_new_game(reload_teams=reload_teams)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/game/restart")
    def restart_game() -> Dict[str, Any]:
        try:
            state = session.start_new_game(reload_teams=True)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/game/stop")
    def stop_game() -> Dict[str, Any]:
        state = session.stop_game()
        return jsonify(state)

    @api_bp.post("/game/swing")
    def swing() -> Dict[str, Any]:
        try:
            state = session.execute_normal_play()
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/game/bunt")
    def bunt() -> Dict[str, Any]:
        try:
            state = session.execute_bunt()
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/strategy/pinch_hit")
    def pinch_hit() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        lineup_index = parse_int_param(payload, "lineup_index")
        bench_index = parse_int_param(payload, "bench_index")
        try:
            state = session.execute_pinch_hit(lineup_index=lineup_index, bench_index=bench_index)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/strategy/defense_substitution")
    def defensive_substitution() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        swaps = payload.get("swaps")
        force = bool(payload.get("force"))
        if isinstance(swaps, list):
            try:
                state = session.execute_defensive_substitution(swaps=swaps, force_illegal=force)
            except GameSessionError as exc:
                return create_error_response(str(exc), session)
        else:
            lineup_index = parse_int_param(payload, "lineup_index")
            bench_index = parse_int_param(payload, "bench_index")
            try:
                state = session.execute_defensive_substitution(
                    lineup_index=lineup_index,
                    bench_index=bench_index,
                    force_illegal=force,
                )
            except GameSessionError as exc:
                return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/strategy/change_pitcher")
    def change_pitcher() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        pitcher_index = parse_int_param(payload, "pitcher_index")
        try:
            state = session.execute_pitcher_change(pitcher_index=pitcher_index)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/log/clear")
    def clear_log() -> Dict[str, Any]:
        state = session.clear_log()
        return jsonify(state)

    @api_bp.post("/teams/reload")
    def reload_teams() -> Dict[str, Any]:
        state = session.reload_teams()
        return jsonify(state)

    @api_bp.post("/teams/library/select")
    def select_team_files() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        home_id = payload.get("home") or ""
        away_id = payload.get("away") or ""
        try:
            state = session.update_team_selection(str(home_id), str(away_id))
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.get("/teams/library/<team_id>")
    def get_team_definition(team_id: str) -> Dict[str, Any]:
        try:
            team_data = session.get_team_definition(team_id)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify({"team_id": team_id, "team": team_data})

    @api_bp.post("/teams/library/save")
    def save_team_definition() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        team_id = payload.get("team_id")
        team_payload = payload.get("team")
        if not isinstance(team_payload, dict):
            return create_error_response("チームデータの形式が不正です。", session)
        try:
            saved_id, state = session.save_team_definition(team_id, team_payload)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify({"team_id": saved_id, "state": state})

    @api_bp.post("/teams/library/delete")
    def delete_team_definition() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        team_id = payload.get("team_id")
        if not isinstance(team_id, str) or not team_id.strip():
            return create_error_response("削除するチームIDを指定してください。", session)
        try:
            state = session.delete_team_definition(team_id.strip())
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    # ------------------------- Players API -------------------------

    def _load_players_with_ids() -> tuple[dict, bool, str]:
        """Load players.json and ensure each player has a stable unique id.

        Returns (data, mutated, path) where mutated indicates whether ids were
        added and the file should be saved back.
        """
        players_path = path_manager.get_players_data_path()
        data = FileUtils.safe_json_load(players_path, default={"batters": [], "pitchers": []})
        if not isinstance(data, dict):
            data = {"batters": [], "pitchers": []}
        data.setdefault("batters", [])
        data.setdefault("pitchers", [])

        mutated = False
        for role_key in ("batters", "pitchers"):
            items = data.get(role_key, []) or []
            for p in items:
                if isinstance(p, dict) and not p.get("id"):
                    # assign a UUIDv4
                    p["id"] = str(uuid.uuid4())
                    mutated = True
        return data, mutated, players_path

    def _save_players_data(data: dict, path: str) -> bool:
        return FileUtils.safe_json_save(data, path)

    def _find_player_by_id(data: dict, player_id: str) -> tuple[Optional[dict], Optional[str], Optional[int]]:
        """Find player by id across both roles.

        Returns (player_ref, role_key, index) where player_ref is the dict in place.
        """
        if not player_id:
            return None, None, None
        for role_key in ("batters", "pitchers"):
            items: List[dict] = data.get(role_key, []) or []
            for idx, p in enumerate(items):
                if isinstance(p, dict) and p.get("id") == player_id:
                    return p, role_key, idx
        return None, None, None

    def _infer_role_from_player_dict(player: dict, fallback: str = "batter") -> str:
        role = (request.get_json(silent=True) or {}).get("role") if False else None  # placeholder
        # We won't read request here; rely on content
        if not isinstance(player, dict):
            return fallback
        # crude inference: presence of pitcher fields indicates pitcher
        if "stamina" in player or "pitcher_type" in player:
            return "pitcher"
        return "batter"

    def _team_mentions_player(team_obj: dict, target_name: str) -> bool:
        if not isinstance(team_obj, dict) or not target_name:
            return False
        # pitchers: list of names
        for pitcher_name in team_obj.get('pitchers', []) or []:
            if isinstance(pitcher_name, str) and pitcher_name == target_name:
                return True
        # lineup: list of {name, position}
        for entry in team_obj.get('lineup', []) or []:
            if isinstance(entry, dict) and entry.get('name') == target_name:
                return True
        # bench: list of names
        for bench_name in team_obj.get('bench', []) or []:
            if isinstance(bench_name, str) and bench_name == target_name:
                return True
        return False

    def _find_referencing_teams(target_name: str) -> list[str]:
        refs: list[str] = []
        if not target_name:
            return refs
        # Check aggregated teams.json (home_team/away_team)
        try:
            teams_path = path_manager.get_teams_data_path()
            teams_data = FileUtils.safe_json_load(teams_path, default=None)
            if isinstance(teams_data, dict):
                for key in ("home_team", "away_team"):
                    team_obj = teams_data.get(key)
                    if isinstance(team_obj, dict) and _team_mentions_player(team_obj, target_name):
                        team_name = team_obj.get('name') or key
                        refs.append(str(team_name))
        except Exception:
            pass

        # Check team library files
        try:
            teams_dir = path_manager.get_team_library_path()
            if os.path.isdir(teams_dir):
                for fname in os.listdir(teams_dir):
                    if not fname.lower().endswith('.json'):
                        continue
                    fpath = os.path.join(teams_dir, fname)
                    team_obj = FileUtils.safe_json_load(fpath, default=None)
                    if isinstance(team_obj, dict) and _team_mentions_player(team_obj, target_name):
                        team_name = team_obj.get('name') or os.path.splitext(fname)[0]
                        refs.append(str(team_name))
        except Exception:
            pass

        return refs
    @api_bp.get("/players/list")
    def list_players() -> Dict[str, Any]:
        role = (request.args.get('role') or 'batter').strip().lower()
        data, mutated, players_path = _load_players_with_ids()
        if mutated:
            _save_players_data(data, players_path)

        def pack(items: list[dict]) -> list[dict]:
            out = []
            for p in items:
                if not isinstance(p, dict):
                    continue
                name = p.get('name')
                pid = p.get('id')
                if name and pid:
                    out.append({'id': str(pid), 'name': str(name)})
            return out

        if role == 'pitcher':
            return jsonify({"players": pack(list(data.get('pitchers', [])))})
        return jsonify({"players": pack(list(data.get('batters', [])))})

    @api_bp.get("/players/detail")
    def player_detail() -> Dict[str, Any]:
        player_id = (request.args.get('id') or '').strip()
        name = (request.args.get('name') or '').strip()
        data, mutated, players_path = _load_players_with_ids()
        if mutated:
            _save_players_data(data, players_path)

        if player_id:
            p, role_key, _ = _find_player_by_id(data, player_id)
            if p and role_key:
                referenced_by = _find_referencing_teams(p.get('name'))
                return jsonify({
                    "player": p,
                    "role": 'pitcher' if role_key == 'pitchers' else 'batter',
                    "referenced_by": referenced_by,
                    "has_references": bool(referenced_by),
                })
            return create_error_response("指定された選手IDは見つかりません。", session)

        if not name:
            return create_error_response("選手IDまたは選手名を指定してください。", session)

        for role_key in ("batters", "pitchers"):
            for p in data.get(role_key, []):
                if isinstance(p, dict) and p.get('name') == name:
                    referenced_by = _find_referencing_teams(p.get('name'))
                    return jsonify({
                        "player": p,
                        "role": 'pitcher' if role_key == 'pitchers' else 'batter',
                        "referenced_by": referenced_by,
                        "has_references": bool(referenced_by),
                    })
        return create_error_response("指定された選手は見つかりません。", session)

    @api_bp.post("/players/save")
    def player_save() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        player = payload.get('player')
        role = (payload.get('role') or '').strip().lower()
        original_name = (payload.get('original_name') or '').strip()
        player_id = (payload.get('player_id') or payload.get('id') or (player or {}).get('id') or '').strip() if isinstance(payload.get('player') or {}, dict) else (payload.get('player_id') or payload.get('id') or '').strip()
        if not isinstance(player, dict) or not player.get('name'):
            return create_error_response("選手データの形式が不正です。", session)
        if role not in ('batter', 'pitcher'):
            # 推定する：pitcher必須キーがあれば投手扱い
            role = 'pitcher' if 'stamina' in player or 'pitcher_type' in player else 'batter'

        data, mutated, players_path = _load_players_with_ids()

        # When updating, prefer player_id; fall back to original_name for backward compatibility
        existing_player = None
        existing_role_key = None
        existing_index: Optional[int] = None
        if player_id:
            existing_player, existing_role_key, existing_index = _find_player_by_id(data, player_id)
        elif original_name:
            for rk in ('batters', 'pitchers'):
                items: List[dict] = data.get(rk, []) or []
                for idx, it in enumerate(items):
                    if isinstance(it, dict) and it.get('name') == original_name:
                        existing_player, existing_role_key, existing_index = it, rk, idx
                        break
                if existing_player:
                    break

        # Ensure the player has an id (reuse existing if present)
        if existing_player and existing_player.get('id'):
            player['id'] = existing_player['id']
        elif player_id:
            # trust provided id if any
            player['id'] = player_id
        else:
            # new player
            player.setdefault('id', str(uuid.uuid4()))

        # Determine target list key based on role
        target_key = 'pitchers' if role == 'pitcher' else 'batters'

        if existing_player is not None and existing_role_key is not None and existing_index is not None:
            # Replace existing at index (may also move between roles if role changed)
            # Remove old entry
            del data[existing_role_key][existing_index]
            # Append to target role (no name-based stripping)
            data[target_key].append(player)
        else:
            # Create new entry; do not strip same-name players
            data[target_key].append(player)

        # Save file if needed
        if mutated or True:
            if not _save_players_data(data, players_path):
                return create_error_response("選手データの保存に失敗しました。", session)

        # 保存後、stateを返してUIを再同期
        state = session.build_state()
        return jsonify({"id": player['id'], "name": player['name'], "role": role, "state": state})

    @api_bp.post("/players/delete")
    def player_delete() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        player_id = (payload.get('player_id') or payload.get('id') or '').strip()
        name = (payload.get('name') or '').strip()
        role = (payload.get('role') or '').strip().lower()

        data, mutated, players_path = _load_players_with_ids()

        # Resolve the player's display name (for reference checks)
        player_name_for_check = None
        if player_id:
            p, _, _ = _find_player_by_id(data, player_id)
            if p and isinstance(p, dict):
                player_name_for_check = p.get('name') or name or None
        if not player_name_for_check:
            player_name_for_check = name or None

        # Prevent deletion if referenced by any team (checked by name)
        referencing = _find_referencing_teams(player_name_for_check)
        if referencing:
            team_list = ", ".join(referencing[:5]) + (" 他" + str(len(referencing) - 5) + "件" if len(referencing) > 5 else "")
            return create_error_response(
                f"選手 '{player_name_for_check}' はチーム({team_list})に含まれているため削除できません。先に該当チームから外してください。",
                session,
            )

        removed_any = False
        if player_id:
            # precise removal by id
            p, role_key, idx = _find_player_by_id(data, player_id)
            if p is not None and role_key is not None and idx is not None:
                del data[role_key][idx]
                removed_any = True
        else:
            if not name:
                return create_error_response("削除する選手IDまたは選手名を指定してください。", session)

            def remove_by_name(items, target_name):
                removed = False
                output = []
                for p in items:
                    if isinstance(p, dict) and p.get('name') == target_name:
                        removed = True
                    else:
                        output.append(p)
                return removed, output

            if role in ('batter', 'pitcher'):
                key = 'pitchers' if role == 'pitcher' else 'batters'
                removed_any, data[key] = remove_by_name(list(data[key]), name)
            else:
                removed_bat, new_bat = remove_by_name(list(data['batters']), name)
                removed_pit, new_pit = remove_by_name(list(data['pitchers']), name)
                data['batters'] = new_bat
                data['pitchers'] = new_pit
                removed_any = removed_bat or removed_pit

        if not removed_any:
            return create_error_response("指定された選手は見つかりません。", session)

        if not _save_players_data(data, players_path):
            return create_error_response("選手データの保存に失敗しました。", session)

        # Note: Teams that reference this player are not auto-updated here.
        # Frontend can warn users or they can update team JSON accordingly.
        state = session.build_state()
        return jsonify({"deleted": player_id or name, "state": state})

    @api_bp.get("/health")
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"})

    return api_bp