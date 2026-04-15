"""
Shared pytest fixtures for sibi_tools.core tests.
"""
import datetime
import sqlite3

import pytest
import shutil
from pathlib import Path
from tempfile import mkdtemp

# Python 3.12+ deprecated the implicit sqlite3 date/datetime adapters.
# Register explicit ISO-string adapters so SQLAlchemy tests using SQLite
# with datetime.date / datetime.datetime bind parameters don't emit warnings.
sqlite3.register_adapter(datetime.date, lambda d: d.isoformat())
sqlite3.register_adapter(datetime.datetime, lambda dt: dt.isoformat())


@pytest.fixture
def temp_project_root():
    """Creates a temporary directory acting as a project root."""
    path = Path(mkdtemp())
    # Add a marker to ensure ProjectService detects it correctly
    (path / "pyproject.toml").touch()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def temp_log_dir():
    """Creates a temporary logging directory."""
    path = Path(mkdtemp())
    yield path
    shutil.rmtree(path)
