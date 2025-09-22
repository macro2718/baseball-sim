"""Flask application exposing the baseball simulator to the browser."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from .session import GameSessionError, WebGameSession

APP_ROOT = Path(__file__).resolve().parent


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        template_folder=str(APP_ROOT / "templates"),
        static_folder=str(APP_ROOT / "static"),
    )
    session = WebGameSession()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/game/state")
    def get_state() -> Dict[str, Any]:
        state = session.build_state()
        return jsonify(state)

    @app.post("/api/game/start")
    def start_game() -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}
        reload_teams = bool(payload.get("reload", False))
        try:
            state = session.start_new_game(reload_teams=reload_teams)
        except GameSessionError as exc:
            return jsonify({"error": str(exc), "state": session.build_state()}), 400
        return jsonify(state)

    @app.post("/api/game/restart")
    def restart_game() -> Dict[str, Any]:
        try:
            state = session.start_new_game(reload_teams=True)
        except GameSessionError as exc:
            return jsonify({"error": str(exc), "state": session.build_state()}), 400
        return jsonify(state)

    @app.post("/api/game/stop")
    def stop_game() -> Dict[str, Any]:
        state = session.stop_game()
        return jsonify(state)

    @app.post("/api/game/swing")
    def swing() -> Dict[str, Any]:
        try:
            state = session.execute_normal_play()
        except GameSessionError as exc:
            return jsonify({"error": str(exc), "state": session.build_state()}), 400
        return jsonify(state)

    @app.post("/api/game/bunt")
    def bunt() -> Dict[str, Any]:
        try:
            state = session.execute_bunt()
        except GameSessionError as exc:
            return jsonify({"error": str(exc), "state": session.build_state()}), 400
        return jsonify(state)

    @app.post("/api/log/clear")
    def clear_log() -> Dict[str, Any]:
        state = session.clear_log()
        return jsonify(state)

    @app.post("/api/teams/reload")
    def reload_teams() -> Dict[str, Any]:
        state = session.reload_teams()
        return jsonify(state)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"})

    return app


app = create_app()
