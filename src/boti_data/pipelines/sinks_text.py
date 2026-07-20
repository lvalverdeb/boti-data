"""CSV and JSONL pipeline sinks.

Split out of sinks.py purely for line-count headroom. Re-exported from
sinks.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import dask.dataframe as dd
import fsspec
import pandas as pd
from boti.core import ResourceConfig, SecureResource
from boti_dask import safe_persist
from pydantic import Field, field_validator

from boti_data.pipelines.sinks_common import (
    FrameResult,
    SinkWriteResult,
    _AsyncWriteViaThreadMixin,
    _validate_storage_path,
    _write_with_staging,
    prepare_partitioned_frame,
    to_dask_frame,
)


class CsvSinkConfig(ResourceConfig):
    """Validated configuration for directory-backed CSV sink writes."""

    storage_path: str
    filename_pattern: str = Field(default="part-*.csv")
    partition_on: list[str] | None = None

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        return _validate_storage_path(value)

    @field_validator("filename_pattern")
    @classmethod
    def validate_filename_pattern(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("filename_pattern must be a simple base name pattern, not a path.")
        if "*" not in value:
            raise ValueError("filename_pattern must include '*' so Dask can shard CSV outputs.")
        return value


class CsvSink(_AsyncWriteViaThreadMixin, SecureResource):
    """Write a frame to a CSV dataset directory, optionally partitioned by one column."""

    def __init__(
        self,
        config: CsvSinkConfig | Mapping[str, Any],
        *,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        resolved_config = (
            config if isinstance(config, CsvSinkConfig) else CsvSinkConfig(**dict(config))
        )
        super().__init__(config=resolved_config, fs=fs)
        self.config = resolved_config

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        """Write *frame* to the configured CSV destination.

        ``overwrite=True`` replaces the *entire* target directory (all
        partitions/dates), not just the partition(s) present in *frame* — see
        :func:`_write_with_staging`. For incremental per-date writes, do not
        rely on ``overwrite=True`` to mean "overwrite this date only".
        """
        ddf = to_dask_frame(frame)
        ddf = prepare_partitioned_frame(
            ddf,
            partition_on=self.config.partition_on,
            date_field=date_field,
            sink_name="CsvSink",
        )
        if persist:
            ddf = safe_persist(ddf)

        fs, fs_path, public_path = self._filesystem_parts()

        def _write(directory: str) -> list[str]:
            if self.config.partition_on:
                return self._write_partitioned_csv(
                    ddf,
                    fs=fs,
                    base_path=directory,
                    write_index=write_index,
                )
            return self._write_csv_pattern(
                ddf,
                fs=fs,
                directory=directory,
                write_index=write_index,
            )

        files = _write_with_staging(
            fs=fs, target_path=fs_path, overwrite=overwrite, write_fn=_write
        )
        return SinkWriteResult(path=public_path, files=tuple(files))

    def _write_partitioned_csv(
        self,
        frame: dd.DataFrame,
        *,
        fs: fsspec.AbstractFileSystem,
        base_path: str,
        write_index: bool,
    ) -> list[str]:
        partition_on = self.config.partition_on or []
        if len(partition_on) != 1:
            raise ValueError("CsvSink currently supports at most one partition column.")
        partition_col = partition_on[0]
        partition_series = frame[partition_col].dropna().astype(str)
        raw_values = partition_series.unique().compute()
        deduped_values = pd.Series(raw_values).dropna().unique()
        values = [str(value) for value in deduped_values.tolist()]
        written: list[str] = []
        for value in values:
            directory = f"{base_path}/{partition_col}={value}".rstrip("/")
            subset = frame[frame[partition_col].astype(str) == value]
            written.extend(
                self._write_csv_pattern(
                    subset,
                    fs=fs,
                    directory=directory,
                    write_index=write_index,
                )
            )
        return sorted(written)

    def _write_csv_pattern(
        self,
        frame: dd.DataFrame,
        *,
        fs: fsspec.AbstractFileSystem,
        directory: str,
        write_index: bool,
    ) -> list[str]:
        fs.makedirs(directory, exist_ok=True)
        pattern = f"{directory.rstrip('/')}/{self.config.filename_pattern}"
        frame.to_csv(pattern, index=write_index)
        return [
            self._restore_protocol(str(path))
            for path in sorted(fs.glob(f"{directory.rstrip('/')}/*.csv"))
        ]

    def _filesystem_parts(self) -> tuple[fsspec.AbstractFileSystem, str, str]:
        storage_path = self.config.storage_path
        parsed = urlparse(storage_path)
        if parsed.scheme and parsed.scheme != "file":
            fs, fs_path = fsspec.core.url_to_fs(storage_path)
            return fs, fs_path.rstrip("/"), storage_path.rstrip("/")

        local_path = parsed.path if parsed.scheme == "file" else storage_path
        if "\x00" in local_path:
            raise ValueError("CSV sink path contains a null byte and has been rejected.")
        secure_path = self.get_secure_path(local_path)
        fs = self.fs or fsspec.filesystem("file")
        return fs, str(secure_path).rstrip("/"), str(secure_path).rstrip("/")

    @staticmethod
    def _restore_protocol(path: str) -> str:
        return str(path)


class JsonlSinkConfig(ResourceConfig):
    """Validated configuration for directory-backed JSONL sink writes."""

    storage_path: str
    filename_pattern: str = Field(default="part-*.jsonl")
    partition_on: list[str] | None = None

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        return _validate_storage_path(value)

    @field_validator("filename_pattern")
    @classmethod
    def validate_filename_pattern(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("filename_pattern must be a simple base name pattern, not a path.")
        if "*" not in value:
            raise ValueError("filename_pattern must include '*' so Dask can shard JSONL outputs.")
        if not value.endswith(".jsonl"):
            raise ValueError("filename_pattern must end with '.jsonl'.")
        return value


class JsonlSink(_AsyncWriteViaThreadMixin, SecureResource):
    """Write a frame to a JSONL dataset directory, optionally partitioned by one column."""

    def __init__(
        self,
        config: JsonlSinkConfig | Mapping[str, Any],
        *,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        resolved_config = (
            config if isinstance(config, JsonlSinkConfig) else JsonlSinkConfig(**dict(config))
        )
        super().__init__(config=resolved_config, fs=fs)
        self.config = resolved_config

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        """Write *frame* to the configured JSONL destination.

        ``overwrite=True`` replaces the *entire* target directory (all
        partitions/dates), not just the partition(s) present in *frame* — see
        :func:`_write_with_staging`. For incremental per-date writes, do not
        rely on ``overwrite=True`` to mean "overwrite this date only".
        """
        ddf = to_dask_frame(frame)
        ddf = prepare_partitioned_frame(
            ddf,
            partition_on=self.config.partition_on,
            date_field=date_field,
            sink_name="JsonlSink",
        )
        if persist:
            ddf = safe_persist(ddf)

        fs, fs_path, public_path = self._filesystem_parts()

        def _write(directory: str) -> list[str]:
            if self.config.partition_on:
                return self._write_partitioned_jsonl(
                    ddf,
                    fs=fs,
                    base_path=directory,
                    write_index=write_index,
                )
            return self._write_jsonl_pattern(
                ddf,
                fs=fs,
                directory=directory,
                write_index=write_index,
            )

        files = _write_with_staging(
            fs=fs, target_path=fs_path, overwrite=overwrite, write_fn=_write
        )
        return SinkWriteResult(path=public_path, files=tuple(files))

    def _write_partitioned_jsonl(
        self,
        frame: dd.DataFrame,
        *,
        fs: fsspec.AbstractFileSystem,
        base_path: str,
        write_index: bool,
    ) -> list[str]:
        partition_on = self.config.partition_on or []
        if len(partition_on) != 1:
            raise ValueError("JsonlSink currently supports at most one partition column.")
        partition_col = partition_on[0]
        partition_series = frame[partition_col].dropna().astype(str)
        raw_values = partition_series.unique().compute()
        deduped_values = pd.Series(raw_values).dropna().unique()
        values = [str(value) for value in deduped_values.tolist()]
        written: list[str] = []
        for value in values:
            directory = f"{base_path}/{partition_col}={value}".rstrip("/")
            subset = frame[frame[partition_col].astype(str) == value]
            written.extend(
                self._write_jsonl_pattern(
                    subset,
                    fs=fs,
                    directory=directory,
                    write_index=write_index,
                )
            )
        return sorted(written)

    def _write_jsonl_pattern(
        self,
        frame: dd.DataFrame,
        *,
        fs: fsspec.AbstractFileSystem,
        directory: str,
        write_index: bool,
    ) -> list[str]:
        fs.makedirs(directory, exist_ok=True)
        pattern = f"{directory.rstrip('/')}/{self.config.filename_pattern}"
        frame.to_json(
            pattern,
            orient="records",
            lines=True,
            date_format="iso",
            compute=True,
        )
        return [
            self._restore_protocol(str(path))
            for path in sorted(fs.glob(f"{directory.rstrip('/')}/*.jsonl"))
        ]

    def _filesystem_parts(self) -> tuple[fsspec.AbstractFileSystem, str, str]:
        storage_path = self.config.storage_path
        parsed = urlparse(storage_path)
        if parsed.scheme and parsed.scheme != "file":
            fs, fs_path = fsspec.core.url_to_fs(storage_path)
            return fs, fs_path.rstrip("/"), storage_path.rstrip("/")

        local_path = parsed.path if parsed.scheme == "file" else storage_path
        if "\x00" in local_path:
            raise ValueError("JSONL sink path contains a null byte and has been rejected.")
        secure_path = self.get_secure_path(local_path)
        fs = self.fs or fsspec.filesystem("file")
        return fs, str(secure_path).rstrip("/"), str(secure_path).rstrip("/")

    @staticmethod
    def _restore_protocol(path: str) -> str:
        return str(path)
