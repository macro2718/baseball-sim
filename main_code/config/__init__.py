"""
Configuration package.
Re-exports common configuration utilities for convenience.
"""

# Re-export frequently used members for simple imports like `from config import X`
from .config import config  # noqa: F401
from .constants import *  # noqa: F401,F403
from .path_utils import path_manager  # noqa: F401
from .project_config import setup_project_environment, get_project_paths  # noqa: F401

