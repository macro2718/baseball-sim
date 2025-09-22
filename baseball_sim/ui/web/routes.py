"""API route handlers for the baseball simulator web interface."""

from __future__ import annotations

from typing import Any, Dict, Callable, Tuple

from flask import Blueprint, jsonify, request, Response

from .session import GameSessionError, WebGameSession

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
        lineup_index = parse_int_param(payload, "lineup_index")
        bench_index = parse_int_param(payload, "bench_index")
        try:
            state = session.execute_defensive_substitution(
                lineup_index=lineup_index,
                bench_index=bench_index,
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

    @api_bp.get("/health")
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"})

    return api_bp