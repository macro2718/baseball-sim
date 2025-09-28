"""Flask application exposing the baseball simulator to the browser."""

from __future__ import annotations

import random
from pathlib import Path

from flask import Flask, render_template

from .routes import create_routes
from .session import WebGameSession
from baseball_sim.config import config
from baseball_sim.infrastructure.logging_utils import logger

APP_ROOT = Path(__file__).resolve().parent


_REQUIRED_CONFIG_KEYS = (
    "game.max_innings",
    "simulation.use_ml_prediction",
    "files.data_dir",
)


def _initialize_application() -> None:
    """Prepare logging, random seed and sanity-check configuration values."""

    logger.info("Starting application initialization...")

    missing_keys = [key for key in _REQUIRED_CONFIG_KEYS if config.get(key) is None]
    for key in missing_keys:
        logger.warning("Missing required config: %s", key)

    seed = config.get("simulation.random_seed")
    if seed is not None:
        random.seed(seed)
        logger.info("Random seed set to: %s", seed)
    else:
        logger.info("Random seed not fixed (using system randomness)")

    logger.info("Application initialization completed successfully")


def create_app() -> Flask:
    """Create and configure the Flask application."""

    # Ensure application initialization (logging, optional seed)
    try:
        _initialize_application()
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
