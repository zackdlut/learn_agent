"""Load repository .env once and resolve paths relative to it.

Call ``load_env()`` at process entry (``main``). Other modules should call
``ensure_env_loaded()`` before reading settings from ``os.environ``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_loaded = False
_env_file: Path | None = None


def find_env_file() -> Path | None:
    """Walk upward from this package directory until a ``.env`` file is found."""
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(override: bool = True) -> Path | None:
    """Load variables from the repo ``.env`` into ``os.environ``."""
    global _loaded, _env_file
    _env_file = find_env_file()
    if _env_file is not None:
        load_dotenv(_env_file, override=override)
    _loaded = True
    return _env_file


def ensure_env_loaded() -> None:
    """Load ``.env`` on first use if ``load_env`` has not run yet."""
    if not _loaded:
        load_env(override=True)


def env_file() -> Path | None:
    """Return the ``.env`` path used by the last ``load_env`` / ``ensure_env_loaded``."""
    ensure_env_loaded()
    return _env_file


def getenv(name: str, default: str | None = None) -> str | None:
    """Read an environment variable after ensuring ``.env`` is loaded."""
    ensure_env_loaded()
    return os.getenv(name, default)
