"""Exception types used by the web user interface layer."""


class GameSessionError(RuntimeError):
    """Raised when an action cannot be performed in the current session state."""

