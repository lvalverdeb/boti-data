# Examples

Run examples from the repository root with:

```bash
python examples/sql_settings.py
python examples/sql_sync_query_only.py
python examples/sql_sync_writable.py
python examples/sql_model_registry.py
python examples/sql_model_builder.py
python examples/sql_async_resource.py
python examples/sql_partitioned_loader.py
python examples/parquet_resource.py
python examples/data_facade_db.py
python examples/data_facade_parquet.py
python examples/data_facade_diagnostics.py
python examples/data_facade_distributed.py
python examples/data_facade_distributed_parquet.py
python examples/data_facade_datacube.py
python examples/data_facade_datacube_contract_rejection.py
python examples/data_csv_sink_pipeline.py
python examples/data_jsonl_sink_pipeline.py
python examples/data_enrichment_v1.py
python examples/data_parquet_pipeline.py
python examples/data_helper_legacy.py
python examples/data_helper_distributed.py
python examples/data_helper_bootstrap.py
python examples/data_hybrid_dataset.py
python examples/data_hybrid_dataset_sql_parquet.py
python examples/data_hybrid_dataset_distributed.py
```

Pure Dask runtime/resilience examples now live in the `boti-dask` repo, e.g.
`../boti-dask/examples/data_facade_dask_resilience.py`.

`data_helper_bootstrap.py` is the deterministic migration path for
`notebooks/99_Bootstrap_legacy.ipynb` and exercises engine views
(`helper.pandas/.polars/.dask`) plus resilient `safe_*` helpers.

Hybrid dataset walkthroughs:

- `examples/data_hybrid_dataset.py` (SQL historical+live split)
- `examples/data_hybrid_dataset_sql_parquet.py` (Parquet historical + SQL live)
- `examples/data_hybrid_dataset_distributed.py` (Dask session + HybridDataset)
- `notebooks/09_hybrid_dataset.ipynb` (interactive notebook flow)

Parquet materialization walkthrough:

- `examples/data_parquet_pipeline.py` (`ParquetPipeline.materialize()` / reload flow)

Sink/plugin walkthroughs:

- `examples/data_csv_sink_pipeline.py` (`SinkPipeline` + `CsvSink` write-only flow)
- `examples/data_jsonl_sink_pipeline.py` (`SinkPipeline` + named `jsonl` sink via registry)

Enrichment walkthrough:

- `examples/data_enrichment_v1.py` (`AsyncFrameEnricher` + `AttachmentSpec`)

See [`docs/BIG_DATA_READYNESS.md`](../docs/BIG_DATA_READYNESS.md) for operating guidance.
Datacube contract guidance: [`docs/DATACUBE_CONTRACT.md`](../docs/DATACUBE_CONTRACT.md).
