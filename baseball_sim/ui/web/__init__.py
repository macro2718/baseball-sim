"""Browser-based user interface for the baseball simulator."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any):
    """Factory wrapper that imports Flask lazily.

    Importing :mod:`baseball_sim.ui.web` used to require Flask immediately,
    which made it impossible to access helper utilities (such as the
    :class:`~baseball_sim.ui.web.session.WebGameSession`) in environments where
    Flask is not installed.  The web UI only needs Flask when the application is
    actually instantiated, so we defer the import until this function is
    executed.
    """

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
