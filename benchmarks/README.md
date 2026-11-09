# Benchmarks

Eight pytest-benchmark cases, all against SQLite file DSNs and local Parquet so
they run on a CI runner with no services. They are marked `perf` and live
outside `tests/`, so the default suite never runs them.

| Case | File | What it measures |
| --- | --- | --- |
| SQL 1M rows to pandas | `test_sql_load_1m_pandas.py` | the PRD headline metric |
| Parquet 1M rows | `test_parquet_load_1m.py` | hive-partitioned dataset to pandas |
| Polars path | `test_polars_return_path.py` | `return_type="polars"` over the same source |
| IN-clause filter | `test_in_clause_filter.py` | `group_id__in=[...]` over 1M rows |
| Partitioned, 8 workers | `test_partitioned_8_workers.py` | lazy load computed on a LocalCluster |
| Streaming batches | `test_streaming_batches.py` | 1M rows as Arrow record batches |
| Parquet sink write | `test_parquet_sink_write.py` | 100k rows through `ParquetSink` |
| Repository CRUD | `test_repository_crud.py` | 1k insert-then-update pairs |

```bash
make bench            # run, print the table, save benchmarks/results/<run>.json
make bench-compare    # run and fail if any mean regresses >20% against the baseline
make bench-baseline   # regenerate benchmarks/baseline/ (done on each release)
```

The committed baseline under `benchmarks/baseline/<platform>/0001_baseline.json`
is regenerated on every release by the maintainer on the release machine; the
`Benchmarks` workflow compares tagged builds against it with
`--benchmark-compare-fail=mean:20%` over at least five rounds.
