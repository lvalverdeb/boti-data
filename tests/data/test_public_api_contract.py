"""Every public name carries a stability label; deprecations warn on access."""

from __future__ import annotations

import warnings

import pytest

import boti_data
from boti_data._contract import Deprecation


def test_every_public_name_is_stable_or_experimental_never_both() -> None:
    stable, experimental = set(boti_data.STABLE), set(boti_data.EXPERIMENTAL)
    assert not stable & experimental
    assert stable | experimental == set(boti_data.__all__)
    assert len(boti_data.STABLE) == len(stable)
    assert len(boti_data.EXPERIMENTAL) == len(experimental)


def test_labels_cover_only_lazily_loaded_names() -> None:
    assert set(boti_data.STABLE) | set(boti_data.EXPERIMENTAL) == set(boti_data._LAZY)


def test_deprecated_name_warns_once_on_first_access(monkeypatch) -> None:
    name = "SqlRepository"
    monkeypatch.delitem(boti_data.__dict__, name, raising=False)
    monkeypatch.setitem(
        boti_data.DEPRECATED, name, Deprecation(removal="2.0", replacement="boti_data.Repo")
    )
    with pytest.warns(DeprecationWarning, match=r"removed in boti-data 2\.0; use boti_data\.Repo"):
        value = getattr(boti_data, name)
    assert value is boti_data.db.sql_repository.SqlRepository
    # Cached by the first access: the second read bypasses __getattr__ and is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(boti_data, name) is value


def test_undeprecated_access_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert boti_data.FieldMap is not None
