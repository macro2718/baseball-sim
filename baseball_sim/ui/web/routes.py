"""API route handlers for the baseball simulator web interface."""

from __future__ import annotations

from typing import Any, Dict, Callable, Tuple

from flask import Blueprint, jsonify, request, Response

from .session import GameSessionError, WebGameSession
from baseball_sim.config.paths import path_manager, FileUtils
import json

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

    # ------------------------- Players API -------------------------
    @api_bp.get("/players/list")
    def list_players() -> Dict[str, Any]:
        role = (request.args.get('role') or 'batter').strip().lower()
        players_path = path_manager.get_players_data_path()
        data = FileUtils.safe_json_load(players_path, default={"batters": [], "pitchers": []})
        if not isinstance(data, dict):
            data = {"batters": [], "pitchers": []}
        batters = [p.get('name') for p in data.get('batters', []) if isinstance(p, dict) and p.get('name')]
        pitchers = [p.get('name') for p in data.get('pitchers', []) if isinstance(p, dict) and p.get('name')]
        if role == 'pitcher':
            return jsonify({"players": pitchers})
        return jsonify({"players": batters})

    @api_bp.get("/players/detail")
    def player_detail() -> Dict[str, Any]:
        name = (request.args.get('name') or '').strip()
        if not name:
            return create_error_response("選手名を指定してください。", session)
        players_path = path_manager.get_players_data_path()
        data = FileUtils.safe_json_load(players_path, default={"batters": [], "pitchers": []})
        if not isinstance(data, dict):
            data = {"batters": [], "pitchers": []}
        for role_key in ("batters", "pitchers"):
            for p in data.get(role_key, []):
                if isinstance(p, dict) and p.get('name') == name:
                    return jsonify({"player": p, "role": 'pitcher' if role_key == 'pitchers' else 'batter'})
        return create_error_response("指定された選手は見つかりません。", session)

    @api_bp.post("/players/save")
    def player_save() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        player = payload.get('player')
        role = (payload.get('role') or '').strip().lower()
        if not isinstance(player, dict) or not player.get('name'):
            return create_error_response("選手データの形式が不正です。", session)
        if role not in ('batter', 'pitcher'):
            # 推定する：pitcher必須キーがあれば投手扱い
            role = 'pitcher' if 'stamina' in player or 'pitcher_type' in player else 'batter'

        players_path = path_manager.get_players_data_path()
        data = FileUtils.safe_json_load(players_path, default={"batters": [], "pitchers": []})
        if not isinstance(data, dict):
            data = {"batters": [], "pitchers": []}
        data.setdefault('batters', [])
        data.setdefault('pitchers', [])

        # 既存を更新 or 追加
        key = 'pitchers' if role == 'pitcher' else 'batters'
        updated = False
        for i, p in enumerate(list(data[key])):
            if isinstance(p, dict) and p.get('name') == player['name']:
                data[key][i] = player
                updated = True
                break
        if not updated:
            data[key].append(player)

        if not FileUtils.safe_json_save(data, players_path):
            return create_error_response("選手データの保存に失敗しました。", session)

        # 保存後、チームに影響は基本なしだが、stateを返してUIを再同期
        state = session.build_state()
        return jsonify({"name": player['name'], "role": role, "state": state})

    @api_bp.get("/health")
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"})

    return api_bp