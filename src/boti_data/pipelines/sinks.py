from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, cast
from urllib.parse import urlparse

import dask.dataframe as dd
import fsspec
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core.models import ResourceConfig
from boti.core.secure_io import SecureResource
from boti_dask import safe_persist
from pydantic import Field, field_validator

from boti_data.parquet import ParquetDataConfig, ParquetReader

FrameResult: TypeAlias = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]
ParquetDestination: TypeAlias = Union[ParquetReader, ParquetDataConfig, Mapping[str, Any]]


def to_dask_frame(frame: FrameResult) -> dd.DataFrame:
    if isinstance(frame, dd.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return dd.from_pandas(frame, npartitions=max(1, len(frame) or 1))
    if isinstance(frame, pa.Table):
        return dd.from_pandas(frame.to_pandas(), npartitions=1)
    if isinstance(frame, pl.DataFrame):
        return dd.from_pandas(frame.to_pandas(), npartitions=1)
    raise TypeError(f"Unsupported frame type for sink writes: {type(frame)!r}")


def prepare_partitioned_frame(
    frame: dd.DataFrame,
    *,
    partition_on: Sequence[str] | None,
    date_field: str | None,
    sink_name: str,
) -> dd.DataFrame:
    if not partition_on:
        return frame

    partition_columns = list(partition_on)
    missing = [name for name in partition_columns if name not in frame.columns]
    if "partition_date" in missing:
        if date_field is None:
            raise ValueError(f"{sink_name} cannot derive partition_date without date_field.")
        if date_field not in frame.columns:
            raise ValueError(
                f"{sink_name} expected date_field={date_field!r} in loaded frame columns "
                f"but only found {list(frame.columns)!r}."
            )
        frame = frame.assign(
            partition_date=dd.to_datetime(frame[date_field], errors="coerce").dt.date.astype(str)
        )
        missing = [name for name in partition_columns if name not in frame.columns]

    if missing:
        raise ValueError(
            f"{sink_name} partition columns must already exist in the loaded frame or be derivable "
            f"from date_field. Missing columns: {missing!r}."
        )
    return frame


@dataclass(slots=True)
class SinkWriteResult:
    """Result returned by pipeline sinks after a write operation."""

    path: str
    files: tuple[str, ...] = ()


class PipelineSink(Protocol):
    """Minimal contract for pipeline write targets."""

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult: ...

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult: ...


class CsvSinkConfig(ResourceConfig):
    """Validated configuration for directory-backed CSV sink writes."""

    storage_path: str
    filename_pattern: str = Field(default="part-*.csv")
    partition_on: list[str] | None = None

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("storage_path must be specified.")
        return normalized.rstrip("/")

    @field_validator("filename_pattern")
    @classmethod
    def validate_filename_pattern(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("filename_pattern must be a simple base name pattern, not a path.")
        if "*" not in value:
            raise ValueError("filename_pattern must include '*' so Dask can shard CSV outputs.")
        return value


class CsvSink(SecureResource):
    """Write a frame to a CSV dataset directory, optionally partitioned by one column."""

    def __init__(
        self,
        config: CsvSinkConfig | Mapping[str, Any],
        *,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        resolved_config = config if isinstance(config, CsvSinkConfig) else CsvSinkConfig(**dict(config))
        super().__init__(config=resolved_config, fs=fs)
        self.config = resolved_config

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        return await asyncio.to_thread(
            self.write,
            frame,
            date_field=date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
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
        if overwrite and fs.exists(fs_path):
            try:
                fs.rm(fs_path, recursive=True)
            except TypeError:
                fs.rm(fs_path)

        if self.config.partition_on:
            files = self._write_partitioned_csv(
                ddf,
                fs=fs,
                base_path=fs_path,
                write_index=write_index,
            )
        else:
            files = self._write_csv_pattern(
                ddf,
                fs=fs,
                directory=fs_path,
                write_index=write_index,
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
        raw_values = frame[partition_col].dropna().astype(str).unique().compute()
        values = [str(value) for value in pd.Series(raw_values).dropna().unique().tolist()]
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
        normalized = value.strip()
        if not normalized:
            raise ValueError("storage_path must be specified.")
        return normalized.rstrip("/")

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


class JsonlSink(SecureResource):
    """Write a frame to a JSONL dataset directory, optionally partitioned by one column."""

    def __init__(
        self,
        config: JsonlSinkConfig | Mapping[str, Any],
        *,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        resolved_config = config if isinstance(config, JsonlSinkConfig) else JsonlSinkConfig(**dict(config))
        super().__init__(config=resolved_config, fs=fs)
        self.config = resolved_config

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        return await asyncio.to_thread(
            self.write,
            frame,
            date_field=date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
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
        if overwrite and fs.exists(fs_path):
            try:
                fs.rm(fs_path, recursive=True)
            except TypeError:
                fs.rm(fs_path)

        if self.config.partition_on:
            files = self._write_partitioned_jsonl(
                ddf,
                fs=fs,
                base_path=fs_path,
                write_index=write_index,
            )
        else:
            files = self._write_jsonl_pattern(
                ddf,
                fs=fs,
                directory=fs_path,
                write_index=write_index,
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
        raw_values = frame[partition_col].dropna().astype(str).unique().compute()
        values = [str(value) for value in pd.Series(raw_values).dropna().unique().tolist()]
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


class ParquetSink:
    """Write a frame to a parquet dataset directory and expose the reader used for reloads."""

    def __init__(
        self,
        destination: ParquetDestination,
        *,
        partition_on: Sequence[str] | None = ("partition_date",),
    ) -> None:
        self.partition_on = list(partition_on) if partition_on is not None else None
        self.reader = self._coerce_destination(destination)
        if self.reader.parquet_filename is not None:
            raise ValueError(
                "ParquetSink destination must point to a parquet dataset directory; "
                "parquet_filename is not supported for materialized write targets."
            )

    def __enter__(self) -> ParquetSink:
        self.reader.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.reader.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> ParquetSink:
        await self.reader.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.reader.__aexit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        self.reader.close()

    async def aclose(self) -> None:
        await self.reader.aclose()

    @staticmethod
    def _coerce_destination(destination: ParquetDestination) -> ParquetReader:
        if isinstance(destination, ParquetReader):
            return destination
        if isinstance(destination, ParquetDataConfig):
            return ParquetReader(destination)
        if isinstance(destination, Mapping):
            payload = dict(destination)
            payload.setdefault("backend", "parquet")
            return ParquetReader(payload)
        raise TypeError(
            "ParquetSink destination must be a ParquetReader, ParquetDataConfig, or mapping. "
            f"Got {type(destination)!r}."
        )

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        return await asyncio.to_thread(
            self.write,
            frame,
            date_field=date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        ddf = to_dask_frame(frame)
        ddf = prepare_partitioned_frame(
            ddf,
            partition_on=self.partition_on,
            date_field=date_field,
            sink_name="ParquetSink",
        )
        if persist:
            ddf = safe_persist(ddf)

        fs = self.reader.resource.require_fs()
        target_path = cast(str, self.reader.parquet_storage_path)
        if overwrite and fs.exists(target_path):
            try:
                fs.rm(target_path, recursive=True)
            except TypeError:
                fs.rm(target_path)

        params: dict[str, Any] = {
            "path": target_path,
            "engine": "pyarrow",
            "write_index": write_index,
            "filesystem": fs,
        }
        if self.partition_on:
            params["partition_on"] = list(self.partition_on)
        ddf.to_parquet(**params)

        files = sorted(self._restore_protocol(path) for path in fs.glob(f"{target_path.rstrip('/')}/*"))
        return SinkWriteResult(path=target_path, files=tuple(files))

    @staticmethod
    def _restore_protocol(path: str) -> str:
        return str(path)


__all__ = [
    "FrameResult",
    "CsvSink",
    "CsvSinkConfig",
    "JsonlSink",
    "JsonlSinkConfig",
    "ParquetSink",
    "PipelineSink",
    "SinkWriteResult",
    "prepare_partitioned_frame",
    "to_dask_frame",
]
