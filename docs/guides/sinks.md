# Sinks and enrichment

## SinkPipeline and sinks

Phase 3 adds a minimal sink/plugin layer next to `ParquetPipeline`.

- `SinkPipeline` orchestrates `DataHelper` / `HybridDataset` loads into any sink implementing the pipeline sink contract
- `CsvSink` and `JsonlSink` are write-only dataset sinks (`.csv` / `.jsonl` shards)
- `ParquetPipeline` now uses the same sink abstraction internally via `ParquetSink`

Use the sink registry for named sink creation:

```python
from boti_data import SinkPipeline, available_sinks

assert "jsonl" in available_sinks()

pipeline = SinkPipeline(
    helper,
    "jsonl",
    sink_config={
        "storage_path": "/tmp/source_events_jsonl",
        "partition_on": ["partition_date"],
    },
    date_field="event_date",
)
result = pipeline.write(filters={"status__exact": "active"})
```

```python
from boti_data import CsvSink, DataHelper, SinkPipeline

helper = DataHelper(
    backend="sqlalchemy",
    connection_url=f"sqlite:///{db_path}",
    poolclass="sqlalchemy.pool.NullPool",
    query_only=False,
    table="source_events",
)

sink = CsvSink(
    {
        "storage_path": "/tmp/source_events_csv",
        "partition_on": ["partition_date"],
    }
)

pipeline = SinkPipeline(helper, sink, date_field="event_date")
result = pipeline.write(filters={"status__exact": "active"})

assert result.files
```

`CsvSink`/`JsonlSink` are intentionally write-only in this release. Use `ParquetPipeline` when you need
write + reload orchestration. See `examples/data_csv_sink_pipeline.py` and
`examples/data_jsonl_sink_pipeline.py` for runnable examples.

## Enrichment v1

`AsyncFrameEnricher` adds declarative attachment-based enrichment before downstream writes.
Select attachments by `AttachmentSpec.key` via `cols=[...]`, and enforce bounded unique-value
extraction with `max_unique_values`.

```python
from boti_data import AsyncFrameEnricher, AttachmentSpec

enricher = AsyncFrameEnricher(
    [
        AttachmentSpec(
            key="customer_segment",
            required_cols={"customer_id"},
            attachment_fn=customer_segment_attachment,
            col_to_kwarg={"customer_id": "ids"},
            left_on=["customer_id"],
            right_on=["id"],
            drop_cols=["id"],
            max_unique_values=5000,
        )
    ]
)
enriched = await enricher.aenrich(base_frame, cols=["customer_segment"])
```

`SinkPipeline` accepts `enricher=...` and optional `enrich_cols=[...]` to apply enrichment right
before sink writes.
