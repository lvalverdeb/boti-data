from __future__ import annotations

from types import SimpleNamespace

import pytest

import boti_data.connection_catalog as connection_catalog


def test_local_s3_profile_accepts_legacy_retry_kwargs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    config = SimpleNamespace(
        fs_type="s3",
        fs_path="analytics-bucket/raw/events",
        storage_path="s3://analytics-bucket/raw/events",
    )

    def fake_from_env_prefix(cls, prefix, *, env_file=None, **overrides):
        captured["prefix"] = prefix
        captured["env_file"] = env_file
        captured["overrides"] = overrides
        return config

    class FakeAdapter:
        def __init__(self, adapter_config):
            captured["adapter_config"] = adapter_config
            self.storage_path = adapter_config.storage_path

        def get_filesystem(self):
            return object()

        def invalidate(self):
            captured["invalidated"] = True

    monkeypatch.setattr(
        connection_catalog.FilesystemConfig,
        "from_env_prefix",
        classmethod(fake_from_env_prefix),
    )
    monkeypatch.setattr(connection_catalog, "FilesystemAdapter", FakeAdapter)

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


def test_local_s3_profile_rejects_non_s3_filesystems(monkeypatch, tmp_path):
    def fake_from_env_prefix(cls, prefix, *, env_file=None, **overrides):
        return SimpleNamespace(fs_type="file", fs_path="/tmp/data")

    monkeypatch.setattr(
        connection_catalog.FilesystemConfig,
        "from_env_prefix",
        classmethod(fake_from_env_prefix),
    )

    with pytest.raises(ValueError, match="expected 's3' or 's3a'"):
        connection_catalog.LocalS3Profile("ETL_", env_file=tmp_path / ".env")
