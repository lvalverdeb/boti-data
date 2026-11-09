--8<-- "benchmarks/README.md"

## Current baseline

Medians from the committed baseline (Apple Silicon, CPython 3.13, five rounds or more):

| Case | Median |
| --- | --- |
| Parquet 1M rows to pandas | 9 ms |
| Parquet sink write, 100k rows | 110 ms |
| IN-clause filter to pandas | 602 ms |
| Repository 1k upserts | 984 ms |
| Streaming 1M rows in Arrow batches | 1.18 s |
| SQL 1M rows to pandas | 1.38 s |
| SQL 1M rows to polars | 1.66 s |
| Partitioned load on 8 workers | 5.26 s |
