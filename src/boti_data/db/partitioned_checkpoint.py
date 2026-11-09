"""Resumable partition plans.

A :class:`PlanCheckpoint` records a :class:`SqlPartitionPlan` and, per
partition, whether its rows have been materialised. Each finished partition's
frame is kept as an Arrow/Parquet file in a :class:`CheckpointStore`, so a load
that stops after N partitions can be re-run with ``resume=True`` and fetch only
the remaining ones: finished partitions are read back from the store and the
final frame is complete.

:class:`FileCheckpointStore` keeps one directory per checkpoint key under a
root directory (``plan.json`` plus ``<partition>.parquet`` files), written
atomically the way :class:`boti_data.watermark.FileWatermarkStore` writes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from boti_data.db.partitioned_types import SqlPartitionPlan, SqlPartitionSpec

__all__ = ["CheckpointStore", "FileCheckpointStore", "PlanCheckpoint", "CheckpointedFetch"]

DONE = "done"
PENDING = "pending"


class CheckpointStore(Protocol):
    """Where a checkpoint's plan and finished partition frames live."""

    def read_plan(self, key: str) -> dict[str, Any] | None: ...

    def write_plan(self, key: str, payload: dict[str, Any]) -> None: ...

    def write_partition(self, key: str, partition_key: str, table: pa.Table) -> None: ...

    def done_partitions(self, key: str) -> set[str]: ...

    def read_partition(self, key: str, partition_key: str) -> pa.Table: ...

    def clear(self, key: str) -> None: ...


def _safe_key(key: str) -> str:
    if not key or any(ch in key for ch in "/\\\0") or key in {".", ".."}:
        raise ValueError("checkpoint key must be a simple name, not a path.")
    return key


class FileCheckpointStore:
    """Checkpoints as JSON plus Parquet files under ``root/<key>/``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def _dir(self, key: str) -> Path:
        return self._root / _safe_key(key)

    def read_plan(self, key: str) -> dict[str, Any] | None:
        path = self._dir(key) / "plan.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write_plan(self, key: str, payload: dict[str, Any]) -> None:
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "plan.json.tmp"
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, directory / "plan.json")

    def write_partition(self, key: str, partition_key: str, table: pa.Table) -> None:
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"{_safe_key(partition_key)}.parquet.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, directory / f"{partition_key}.parquet")

    def done_partitions(self, key: str) -> set[str]:
        """Keys of the partitions whose frame is stored (the Parquet file is the commit)."""
        return {path.stem for path in self._dir(key).glob("*.parquet")}

    def read_partition(self, key: str, partition_key: str) -> pa.Table:
        return pq.read_table(self._dir(key) / f"{_safe_key(partition_key)}.parquet")

    def clear(self, key: str) -> None:
        import shutil

        shutil.rmtree(self._dir(key), ignore_errors=True)


@dataclass
class PlanCheckpoint:
    """A plan whose partitions are done once their frame sits in the store.

    Done-ness is derived from the store (the Parquet file written atomically
    by :meth:`mark_done` is the commit), so copies of a checkpoint in several
    Dask tasks or processes never disagree and never overwrite each other.
    """

    store: CheckpointStore
    key: str
    plan: SqlPartitionPlan | None = None

    @classmethod
    def load(cls, store: CheckpointStore, key: str) -> PlanCheckpoint | None:
        """The checkpoint stored under ``key``, or ``None`` when there is none."""
        payload = store.read_plan(key)
        if payload is None:
            return None
        return cls(store=store, key=key, plan=SqlPartitionPlan.from_json(payload["plan"]))

    def begin(self, plan: SqlPartitionPlan) -> None:
        """Start afresh from ``plan``: stored frames discarded, every partition pending."""
        self.store.clear(self.key)
        self.plan = plan
        self.store.write_plan(self.key, {"plan": plan.to_json()})

    def is_done(self, partition: SqlPartitionSpec) -> bool:
        return partition.key() in self.store.done_partitions(self.key)

    @property
    def status(self) -> dict[str, str]:
        assert self.plan is not None
        done = self.store.done_partitions(self.key)
        return {p.key(): DONE if p.key() in done else PENDING for p in self.plan.partitions}

    @property
    def pending(self) -> tuple[SqlPartitionSpec, ...]:
        assert self.plan is not None
        done = self.store.done_partitions(self.key)
        return tuple(p for p in self.plan.partitions if p.key() not in done)

    @property
    def complete(self) -> bool:
        return self.plan is not None and not self.pending

    def mark_done(self, partition: SqlPartitionSpec, frame: pd.DataFrame) -> None:
        """Persist ``frame`` for ``partition``; the write itself records it as done."""
        table = pa.Table.from_pandas(frame, preserve_index=False)
        self.store.write_partition(self.key, partition.key(), table)

    def stored_frame(self, partition: SqlPartitionSpec) -> pd.DataFrame:
        from boti_data.db.arrow_schema_mapper import arrow_table_to_pandas

        return arrow_table_to_pandas(self.store.read_partition(self.key, partition.key()))


class CheckpointedFetch:
    """A partition fetcher that serves done partitions from the checkpoint and records new ones.

    Keeps the executor's ``fetch_partition`` keyword-only signature, so it drops
    into ``SqlPartitionExecutor.load_plan``; it is a plain picklable object, so
    Dask tasks may carry copies of it safely (see :class:`PlanCheckpoint`).
    """

    def __init__(self, checkpoint: PlanCheckpoint, fetch: Callable[..., pd.DataFrame]) -> None:
        self.checkpoint = checkpoint
        self.fetch = fetch

    def __call__(self, *, partition: SqlPartitionSpec, **kwargs: Any) -> pd.DataFrame:
        if self.checkpoint.is_done(partition):
            return self.checkpoint.stored_frame(partition)
        frame = self.fetch(partition=partition, **kwargs)
        self.checkpoint.mark_done(partition, frame)
        return frame
