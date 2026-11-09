# Incremental loads

Three pieces turn a full reload into an incremental one: a watermark that
remembers how far the last load got, a checkpoint that lets a long partitioned
load resume where it stopped, and an upsert sink that merges the new rows into
a table instead of appending duplicates.

## Loading only what changed: `load(since=...)`

`DataGateway.load` and `aload` accept an opt-in `since=` keyword. Give it the
name of the watermark column and a store, and the gateway loads only rows
above the stored watermark, computes the result, and advances the watermark to
the new maximum before returning the frame.

```python
from boti_data import DataGateway
from boti_data.db import SqlDatabaseConfig
from boti_data.watermark import FileWatermarkStore

store = FileWatermarkStore(".watermarks.json")
config = SqlDatabaseConfig(connection_url="postgresql://.../warehouse")

with DataGateway(config, table="events") as gateway:
    frame = gateway.load(since="updated_at", watermark_store=store, as_pandas=True)
```

The watermark is keyed by a *source* name, which defaults to the configured
table. In structured mode (no `table=`), or to share one key between several
gateways, pass `watermark_source=` or the long form:

```python
frame = gateway.load(since=("updated_at", store, "events-hourly"), as_pandas=True)
```

Two further keywords tune the comparison: `watermark_operator=` (`"gt"` by
default, `"ge"` to include rows equal to the watermark) and
`watermark_initial=` for the first run when nothing is stored yet. The
watermark only advances when the load succeeds, so a failed run is simply
re-run.

### Where the watermark lives

| Store | Use when |
| --- | --- |
| `FileWatermarkStore(path)` | One process, local disk. Atomic rename on write. |
| `FsspecWatermarkStore(fs, path)` | Shared object storage between workers. |
| `SqlWatermarkStore(engine)` | The watermark should live next to the data, in the same database, under the same backups and permissions. |

`SqlWatermarkStore` keeps one row per source in a `boti_watermarks` table
(`source`, `value`, `updated_at`), created on first use. Values are
JSON-encoded exactly as the file stores encode them, so a watermark can be
moved between stores without conversion.

```python
import sqlalchemy as sa
from boti_data.watermark import SqlWatermarkStore

store = SqlWatermarkStore(sa.create_engine("postgresql://.../warehouse"))
```

## Resuming an interrupted partitioned load

A partitioned SQL load fetches its plan partition by partition. Give
`SqlPartitionedLoader.load_request` a `PlanCheckpoint` and every partition's
frame is written to the checkpoint store the moment it is materialised. If the
process dies after partition 37 of 100, re-run with `resume=True` on the
request: the stored plan is reused, the 37 finished partitions are read back
from the store, only the remaining 63 are fetched, and the returned frame is
complete.

```python
from boti_data.db import (
    FileCheckpointStore,
    PlanCheckpoint,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
)

store = FileCheckpointStore("/var/lib/etl/checkpoints")
request = SqlPartitionedLoadRequest(
    statement=select(Event), model=Event, chunk_size=50_000, as_pandas=True, resume=True
)
with SqlPartitionedLoader(config) as loader:
    frame = loader.load_request(request, checkpoint=PlanCheckpoint(store, "events-2026-11"))
```

Without `resume=True` the checkpoint starts afresh: the stored plan and
frames are discarded and a new plan is written. A partition is done when its
Parquet file exists in the store; there is no separate status file to fall out
of step, so copies of the checkpoint held by several Dask tasks never disagree.

`SqlPartitionPlan.to_json()` and `from_json()` serialise the plan itself, so a
plan computed once can be inspected, stored or handed to another process.

`FileCheckpointStore` keeps one directory per checkpoint key under its root,
holding `plan.json` and one `<partition>.parquet` per finished partition.
Implement the `CheckpointStore` protocol (five methods) to keep checkpoints
elsewhere.

## Merging rows with `UpsertSink`

`UpsertSink` writes a frame into a SQL table with insert-or-update semantics
on the given key columns. The target table must exist with a primary key or
unique constraint on those columns.

```python
from boti_data.pipelines import UpsertSink

with UpsertSink({"connection_url": url}, table="items", key_columns=["id"]) as sink:
    sink.write(frame)
```

| Database | Statement |
| --- | --- |
| PostgreSQL, SQLite | `INSERT ... ON CONFLICT (keys) DO UPDATE SET ...` |
| DuckDB | `INSERT OR REPLACE INTO ...` |
| MySQL | `INSERT ... ON DUPLICATE KEY UPDATE ...` |

The dialect is detected from the engine. pandas, Arrow and polars frames are
written in one transaction in chunks of `chunk_rows` (5,000 by default); a
Dask frame is written partition by partition. The sink implements the
`PipelineSink` protocol, so it can be used through `SinkPipeline` and the sink
registry under the name `"upsert"`, with `table` and `key_columns` in the sink
configuration.

Putting the three together gives a pipeline that loads only new rows, survives
an interruption without refetching, and lands the result without duplicates:

```python
frame = gateway.load(since="updated_at", watermark_store=watermarks, as_pandas=True)
sink.write(frame)
```
