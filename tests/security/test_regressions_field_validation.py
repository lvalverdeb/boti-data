"""
Security regression tests: regex ReDoS guards, FieldMap strict mode, and
ParquetDataResource path sandboxing.

Split out of test_regressions.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pytest

from boti_data.field_map import FieldMap
from boti_data.filters.utils import validate_regex_pattern

pytestmark = pytest.mark.security_regression


@pytest.mark.parametrize(
    "pattern",
    [
        "(a+)+b",  # classic nested-quantifier ReDoS
        "(a*)*b",  # nested star
        "([a-z]+)*",  # alternation of quantified group
        "([a-z]{1,5}){",  # quantified group with brace quantifier
    ],
)
def test_validate_regex_pattern_rejects_dangerous_patterns(pattern) -> None:
    """Known catastrophic-backtracking patterns must raise ValueError."""
    with pytest.raises(ValueError, match="nested quantifiers"):
        validate_regex_pattern(pattern)


def test_validate_regex_pattern_rejects_overlong_patterns() -> None:
    with pytest.raises(ValueError, match="too long"):
        validate_regex_pattern("a" * 501)


def test_validate_regex_pattern_rejects_syntactically_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid regex"):
        validate_regex_pattern("[unclosed")


def test_validate_regex_pattern_accepts_safe_patterns() -> None:
    validate_regex_pattern(r"^\d{4}-\d{2}-\d{2}$")
    validate_regex_pattern(r"foo|bar")
    validate_regex_pattern(r"[a-z]+")


def test_field_map_strict_mode_raises_for_unknown_key() -> None:
    fm = FieldMap({"db_col": "semantic_field"}, strict=True)
    with pytest.raises(KeyError, match="unknown_key"):
        fm.to_db("unknown_key")


def test_field_map_strict_mode_passes_known_key() -> None:
    fm = FieldMap({"db_col": "semantic_field"}, strict=True)
    assert fm.to_db("semantic_field") == "db_col"


def test_field_map_non_strict_passes_unknown_key_unchanged() -> None:
    fm = FieldMap({"db_col": "semantic_field"}, strict=False)
    assert fm.to_db("unknown_key") == "unknown_key"


def test_parquet_resource_rejects_path_traversal_at_construction(tmp_path) -> None:
    """A traversal path must be rejected at construction, not at load time."""
    from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

    # Use an absolute path outside the project root and temp dir to ensure the
    # sandbox boundary is actually crossed.
    traversal_path = "/etc/passwd"
    config = ParquetDataConfig(parquet_storage_path=traversal_path)
    with pytest.raises(PermissionError):
        ParquetDataResource(config)


def test_parquet_resource_rejects_null_byte_in_path(tmp_path) -> None:
    """NUL bytes are now rejected by the shared SecureResource.get_secure_path
    fail-closed handling (boti.core.secure_io) rather than a local ad-hoc check,
    so this asserts PermissionError instead of ValueError."""
    from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

    config = ParquetDataConfig(parquet_storage_path=str(tmp_path))
    resource = ParquetDataResource(config)
    with pytest.raises(PermissionError, match="could not be resolved"):
        resource._secure_local_path("/tmp/file\x00.parquet")
