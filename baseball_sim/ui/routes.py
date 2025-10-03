"""API route handlers for the baseball simulator web interface.

Web Player Editor changes:
- Players are now managed by stable unique IDs instead of names to avoid
    overwriting when multiple players share the same name.
- The players.json file will be auto-migrated to include an "id" field for
    each player the first time the Players API is used, if missing.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from flask import Blueprint, Response, jsonify, request

from .player_service import (
    PlayerLibraryError,
    PlayerLibraryService,
    PlayerDataset,
    PlayerSaveResult,
)
from .session import GameSessionError, WebGameSession
from baseball_sim.config import LeagueAverages


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
    """Create and configure API routes with the given session.

    Note: Create a new Blueprint per app instance to avoid route duplication
    across dev reloads or multiple app factories.
    """

    api_bp = Blueprint('api', __name__, url_prefix='/api')
    player_service = PlayerLibraryService()

    def load_players_dataset(auto_save: bool = True) -> PlayerDataset:
        dataset = player_service.load_dataset()
        if auto_save and dataset.mutated:
            player_service.save_dataset(dataset)
        return dataset
    
    @api_bp.get("/game/state")
    def get_state() -> Dict[str, Any]:
        state = session.build_state()
        return jsonify(state)

    @api_bp.get("/league/averages")
    def get_league_averages() -> Dict[str, Any]:
        league = LeagueAverages.load()
        return jsonify({"averages": league.as_dict()})

    @api_bp.post("/game/start")
    def start_game() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        reload_teams = bool(payload.get("reload", False))
        control_mode = payload.get("mode")
        user_team = payload.get("user_team")
        try:
            state = session.start_new_game(
                reload_teams=reload_teams,
                control_mode=control_mode,
                user_team=user_team,
            )
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    # Note: Restart merged into /game/start with { reload: true } on the client side.

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

    @api_bp.post("/game/squeeze")
    def squeeze() -> Dict[str, Any]:
        try:
            state = session.execute_squeeze()
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/game/progress")
    def progress() -> Dict[str, Any]:
        try:
            state = session.execute_cpu_progress()
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/game/analytics")
    def compute_analytics() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        samples = parse_int_param(payload, "samples", 100)
        try:
            state = session.compute_live_analytics(samples)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/strategy/steal")
    def steal() -> Dict[str, Any]:
        try:
            state = session.execute_steal()
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

    @api_bp.post("/strategy/pinch_run")
    def pinch_run() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        base_index = parse_int_param(payload, "base_index")
        bench_index = parse_int_param(payload, "bench_index")
        try:
            state = session.execute_pinch_run(base_index=base_index, bench_index=bench_index)
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

    @api_bp.post("/simulation/run")
    def run_simulation() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        games = parse_int_param(payload, "games")
        league_payload = payload.get("league")
        try:
            if isinstance(league_payload, dict):
                num_games = games if games > 0 else None
                state = session.run_simulation(num_games, league_options=league_payload)
            else:
                state = session.run_simulation(games)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/simulation/match/start")
    def start_simulation_match() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        away_key = str(payload.get("away") or "")
        home_key = str(payload.get("home") or "")
        control_mode = payload.get("mode")
        user_team = payload.get("user_team")
        try:
            state = session.start_simulation_match(
                away_key=away_key,
                home_key=home_key,
                control_mode=control_mode,
                user_team=user_team,
            )
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/teams/reload")
    def reload_teams() -> Dict[str, Any]:
        state = session.reload_teams()
        return jsonify(state)

    @api_bp.post("/teams/lineup")
    def update_lineup() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        team_key = payload.get("team")
        lineup_payload = payload.get("lineup")
        try:
            state = session.update_starting_lineup(str(team_key), lineup_payload)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
        return jsonify(state)

    @api_bp.post("/teams/pitcher")
    def update_starting_pitcher() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        team_key = payload.get("team")
        pitcher_name = payload.get("pitcher")
        try:
            state = session.set_starting_pitcher(str(team_key), pitcher_name)
        except GameSessionError as exc:
            return create_error_response(str(exc), session)
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

    @api_bp.get("/players/catalog")
    def players_catalog() -> Dict[str, Any]:
        dataset = load_players_dataset()
        catalogue = player_service.build_catalogue(dataset)
        return jsonify(catalogue)

    @api_bp.get("/players/list")
    def list_players() -> Dict[str, Any]:
        role = (request.args.get('role') or 'batter').strip().lower()
        folder = (request.args.get('folder') or '').strip()
        dataset = load_players_dataset()
        player_list = player_service.build_list(dataset, role, folder=folder or None)
        return jsonify(player_list)

    @api_bp.get("/players/detail")
    def player_detail() -> Dict[str, Any]:
        player_id = (request.args.get('id') or '').strip()
        name = (request.args.get('name') or '').strip()
        dataset = load_players_dataset()
        try:
            detail = player_service.get_player_detail(
                dataset, player_id=player_id, name=name
            )
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)
        return jsonify(detail)

    @api_bp.post("/players/save")
    def player_save() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        player = payload.get('player')
        role = (payload.get('role') or '').strip().lower()
        original_name = (payload.get('original_name') or '').strip()
        player_dict = player if isinstance(player, dict) else {}
        player_id_value = (
            payload.get('player_id')
            or payload.get('id')
            or player_dict.get('id')
            or ''
        )
        player_id = str(player_id_value).strip()
        dataset = load_players_dataset(auto_save=False)
        try:
            result: PlayerSaveResult = player_service.save_player(
                dataset,
                player,
                role,
                player_id=player_id,
                original_name=original_name,
            )
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)

        if not player_service.save_dataset(dataset):
            return create_error_response("選手データの保存に失敗しました。", session)

        if result.previous_name:
            try:
                if result.previous_name != result.player_name:
                    player_service.rename_player_references(
                        result.previous_name, result.player_name
                    )
            except Exception:
                pass
            session.ensure_teams(force_reload=True)

        state = session.build_state()
        return jsonify(
            {
                "id": result.player_id,
                "name": result.player_name,
                "role": result.role,
                "state": state,
            }
        )

    # ------------------------- Player Folders API -------------------------

    @api_bp.get("/players/folders")
    def players_folders() -> Dict[str, Any]:
        dataset = load_players_dataset()
        folders = player_service.list_folders(dataset)
        return jsonify({"folders": folders})

    @api_bp.post("/players/folders/create")
    def players_folders_create() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        name = (payload.get('name') or '').strip()
        dataset = load_players_dataset(auto_save=False)
        try:
            folders = player_service.add_folder(dataset, name)
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)
        if not player_service.save_dataset(dataset):
            return create_error_response("タグの保存に失敗しました。", session)
        return jsonify({"folders": folders})

    @api_bp.post("/players/folders/delete")
    def players_folders_delete() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        name = (payload.get('name') or '').strip()
        dataset = load_players_dataset(auto_save=False)
        try:
            folders = player_service.delete_folder(dataset, name)
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)
        if not player_service.save_dataset(dataset):
            return create_error_response("タグの削除に失敗しました。", session)
        return jsonify({"folders": folders})

    @api_bp.post("/players/folders/rename")
    def players_folders_rename() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        old = (payload.get('old') or '').strip()
        new = (payload.get('new') or '').strip()
        dataset = load_players_dataset(auto_save=False)
        try:
            folders = player_service.rename_folder(dataset, old, new)
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)
        if not player_service.save_dataset(dataset):
            return create_error_response("タグの名称変更に失敗しました。", session)
        return jsonify({"folders": folders})

    @api_bp.post("/players/delete")
    def player_delete() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        player_id = (payload.get('player_id') or payload.get('id') or '').strip()
        name = (payload.get('name') or '').strip()
        role = (payload.get('role') or '').strip().lower()

        dataset = load_players_dataset(auto_save=False)
        try:
            deleted_identifier = player_service.delete_player(
                dataset, player_id=player_id, name=name, role=role
            )
        except PlayerLibraryError as exc:
            return create_error_response(exc.message, session)

        if not player_service.save_dataset(dataset):
            return create_error_response("選手データの保存に失敗しました。", session)

        state = session.build_state()
        return jsonify({"deleted": deleted_identifier, "state": state})

    return api_bp
