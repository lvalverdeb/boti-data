from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import boti_data.connection_catalog as connection_catalog


class _CapturingAdapter:
    """Fake FilesystemAdapter that records its constructed config into *captured*."""

    def __init__(self, adapter_config, captured: dict[str, object]) -> None:
        captured["adapter_config"] = adapter_config
        self.storage_path = adapter_config.storage_path
        self._captured = captured

    def get_filesystem(self) -> object:
        return object()

    def invalidate(self) -> None:
        self._captured["invalidated"] = True


def _make_fake_from_env_prefix(config, captured: dict[str, object]):
    def fake_from_env_prefix(cls, prefix, *, env_file=None, **overrides) -> Any:
        captured["prefix"] = prefix
        captured["env_file"] = env_file
        captured["overrides"] = overrides
        return config

    return fake_from_env_prefix


def test_local_s3_profile_accepts_legacy_retry_kwargs(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(
        fs_type="s3",
        fs_path="analytics-bucket/raw/events",
        storage_path="s3://analytics-bucket/raw/events",
    )

    monkeypatch.setattr(
        connection_catalog.FilesystemConfig,
        "from_env_prefix",
        classmethod(_make_fake_from_env_prefix(config, captured)),
    )
    monkeypatch.setattr(
        connection_catalog,
        "FilesystemAdapter",
        lambda adapter_config: _CapturingAdapter(adapter_config, captured),
    )

    profile = connection_catalog.LocalS3Profile(
        "ETL_",
        env_file=tmp_path / ".env",
        max_attempts=7,
        retry_base_delay=1.25,
        fs_endpoint="https://minio.local:9000",
    )

    assert profile.prefix == "ETL_"
    assert profile.env_file == tmp_path / ".env"
    assert profile.max_attempts == 7
    assert profile.retry_base_delay == 1.25
    assert profile.storage_path == "s3://analytics-bucket/raw/events"
    assert captured == {
        "prefix": "ETL_",
        "env_file": tmp_path / ".env",
        "overrides": {"fs_endpoint": "https://minio.local:9000"},
        "adapter_config": config,
    }


def test_local_s3_profile_rejects_non_s3_filesystems(monkeypatch, tmp_path) -> None:
    def fake_from_env_prefix(cls, prefix, *, env_file=None, **overrides) -> SimpleNamespace:
        return SimpleNamespace(fs_type="file", fs_path="/tmp/data")

    monkeypatch.setattr(
        connection_catalog.FilesystemConfig,
        "from_env_prefix",
        classmethod(fake_from_env_prefix),
    )

    with pytest.raises(ValueError, match="expected 's3' or 's3a'"):
        connection_catalog.LocalS3Profile("ETL_", env_file=tmp_path / ".env")


def _make_s3_catalog(monkeypatch, tmp_path, *, fs_path: str) -> connection_catalog.S3Catalog:
    config = SimpleNamespace(fs_type="s3", fs_path=fs_path, storage_path=f"s3://{fs_path}")

    def fake_from_env_prefix(cls, prefix, *, env_file=None, **overrides) -> SimpleNamespace:
        return config

    class FakeAdapter:
        def __init__(self, adapter_config):
            self.storage_path = adapter_config.storage_path

        def get_filesystem(self) -> object:
            return object()

    monkeypatch.setattr(
        connection_catalog.FilesystemConfig,
        "from_env_prefix",
        classmethod(fake_from_env_prefix),
    )
    monkeypatch.setattr(connection_catalog, "FilesystemAdapter", FakeAdapter)

    return connection_catalog.S3Catalog("ETL_", env_file=tmp_path / ".env")


def test_resolve_key_allows_nested_path_inside_prefix(monkeypatch, tmp_path) -> None:
    catalog = _make_s3_catalog(monkeypatch, tmp_path, fs_path="data/prod")
    assert catalog._resolve_key("sub/object.parquet") == "data/prod/sub/object.parquet"


def test_resolve_key_blocks_dotdot_traversal_above_prefix(monkeypatch, tmp_path) -> None:
    catalog = _make_s3_catalog(monkeypatch, tmp_path, fs_path="data/prod")
    with pytest.raises(PermissionError, match="Path traversal denied"):
        catalog._resolve_key("../../etc/passwd")


def test_resolve_key_blocks_sibling_prefix_confusion(monkeypatch, tmp_path) -> None:
    """Regression: _resolve_key used to validate the traversal-normalised
    key with `str(resolved).startswith(str(base))`. A relative_path that
    normalises to a sibling key sharing the same string prefix as the
    configured base (e.g. base='data/prod', resolved='data/prod_public/secret')
    passed that check even though 'prod_public' is not inside 'prod' --
    letting a caller escape the configured S3 prefix into an unrelated one.
    is_relative_to() compares path components instead of raw strings and
    correctly rejects it."""
    catalog = _make_s3_catalog(monkeypatch, tmp_path, fs_path="data/prod")
    with pytest.raises(PermissionError, match="Path traversal denied"):
        catalog._resolve_key("../prod_public/secret")
