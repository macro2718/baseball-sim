"""Flask application exposing the baseball simulator to the browser."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template

from .routes import create_routes
from .session import WebGameSession
from baseball_sim.infrastructure.initializer import AppInitializer

APP_ROOT = Path(__file__).resolve().parent


def create_app() -> Flask:
    """Create and configure the Flask application."""

    # Ensure application initialization (logging, optional seed)
    try:
        AppInitializer.initialize_application()
    except Exception:
        # Avoid crashing app creation due to init errors; they are logged
        pass

    app = Flask(
        __name__,
        template_folder=str(APP_ROOT / "templates"),
        static_folder=str(APP_ROOT / "static"),
    )
    
    # Initialize game session
    session = WebGameSession()
    
    # Register API routes
    api_blueprint = create_routes(session)
    app.register_blueprint(api_blueprint)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    return app


app = create_app()
