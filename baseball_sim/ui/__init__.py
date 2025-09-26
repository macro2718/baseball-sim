"""Browser-based user interface for the baseball simulator.

This package provides a Flask-based UI. Flask is only imported when
``create_app`` is called so that modules like :mod:`baseball_sim.ui.session`
can be used without Flask installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any):
    """Factory wrapper that imports Flask lazily.

    The web UI only needs Flask when the application is actually
    instantiated, so we defer the import until this function is executed.
    """

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
