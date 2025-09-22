"""Baseball simulation package."""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def main(*args: Any, **kwargs: Any) -> Any:
    """Entry point wrapper that defers heavy imports until execution.

    Importing :mod:`baseball_sim` previously triggered the GUI bootstrap
    immediately via ``from baseball_sim.app.main import main``.  This eager
    import pulled in optional dependencies such as tkinter, pandas or even the
    machine learning tooling used for offline model training.  The web API only
    needs lightweight helpers, so the eager import prevented the Flask app from
    starting unless every optional dependency was available.

    By resolving the actual ``main`` function lazily we keep backwards
    compatibility—``baseball_sim.main()`` continues to work—while avoiding the
    side effects when the package is imported for non-GUI contexts.
    """

    from baseball_sim.app.main import main as _main  # local import to avoid eager dependency loading
    return _main(*args, **kwargs)
