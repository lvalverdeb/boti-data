"""
Helpers for SQLAlchemy asyncio runtime requirements.
"""

from __future__ import annotations

from importlib.util import find_spec

from sqlalchemy.exc import SQLAlchemyError


def ensure_greenlet_available() -> None:
    """Fail fast with an actionable error when SQLAlchemy asyncio support is incomplete."""
    if find_spec("greenlet") is None:
        raise SQLAlchemyError(
            "Async SQLAlchemy operations require the 'greenlet' package. "
            "Install the project dependencies into the active environment "
            "(for example, run `uv sync`) before using async database resources."
        )
