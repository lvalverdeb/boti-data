# Big-data readiness

Boti is ready for **Dask-first large workloads** today, with some clear boundaries.

## Recommended operating model

1. Use `return_type="dask"` for large reads.
2. Run under an explicit Dask client or `dask_session(...)`.
3. Prefer partitioned SQL reads or parquet-backed datasets.
4. Prune columns and push filters down as early as possible.
5. Use `indexed_left_join(...)` for large distributed joins after aligning join-key dtypes.
6. Persist after expensive repartition/index boundaries when the result will be reused.
7. Materialize only at the edge for exports, samples, or compact aggregates.
8. Use `resilient=True` when you want Dask persistence to go through the resilience layer.
9. Use `apply_recommended_dask_config(...)` for shuffle-heavy workloads.

## What Boti already supports

- partitioned SQL loading through `DataGateway` and `SqlPartitionedLoader`
- lazy Dask execution on local or external distributed schedulers
- explicit session ownership via `DaskSession` / `dask_session(...)`
- schema-aware distributed joins via `indexed_left_join(...)`
- resilient Dask helpers for sync, async, and distributed actions
- opt-in resilient indexed joins via `indexed_left_join(..., resilient=True)`
- opt-in Dask config preset via `apply_recommended_dask_config(...)`
- opt-in diagnostics on gateway loads and joins with `diagnostics=True`
- eager result modes (`pandas`, `arrow`, `polars`) for small-result convenience

## What `diagnostics=True` gives you

For gateway loads:

- requested vs resolved return type
- active Dask client summary
- partition plan summary
- load timing and Dask graph/partition metrics

For joins:

- active client summary
- input/output frame metrics
- timing for indexed joins
- warning when a direct Dask merge is likely to trigger a large shuffle
- warning when `persist=False` may cause repeated recomputation
- graph metrics for lazy Dask gateway loads

## Current production guidance

### SQL-backed workloads

- Prefer file-backed or network-reachable databases for worker access.
- Keep partitioned statements free of `LIMIT`, `OFFSET`, and `ORDER BY`; let the loader manage partition planning.
- Choose explicit `partition_column` / `order_column` when defaults are not ideal.

### Parquet-backed workloads

- Prefer parquet for repeated large analytics and intermediate datasets.
- Organize data so partition pruning is meaningful.
- Use filesystem profiles or project-rooted paths that are visible to the workers.

### Join-heavy workloads

- Normalize join-key dtypes before joining.
- For large joins, set indexes on the join key and join on the index.
- Persist after `set_index(...)` if downstream work will reuse the result.
- Prefer `resilient=True` for long-running or remote distributed joins.

## Readiness status

**Ready now**

- local distributed development
- explicit remote-style scheduler attachment
- partitioned SQL + parquet lazy loading
- indexed distributed joins
- basic diagnostics and workload visibility
- resilience helpers for compute/persist/wait/inspection
- recommended opt-in config profile for shuffle-heavy jobs

**Still maturing**

- broader validation against real remote clusters and object stores
- heavier benchmark coverage for skew, spill, and failure scenarios
- more production cookbook material for deployment-specific tuning

## Minimal example

```python
from dask.distributed import LocalCluster
from sqlalchemy import select

from boti_data import (
    DataGateway,
    SqlDatabaseConfig,
    apply_recommended_dask_config,
    dask_session,
    indexed_left_join,
)

config = SqlDatabaseConfig(connection_url="sqlite:///warehouse.db", query_only=False)

with apply_recommended_dask_config():
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=":0") as cluster:
        with dask_session(scheduler_address=cluster.scheduler_address):
            with DataGateway(config) as gateway:
                left = gateway.load(
                    statement=select(User),
                    model=User,
                    persist=True,
                    resilient=True,
                    diagnostics=True,
                )
                right = gateway.load(
                    statement=select(UserProfile),
                    model=UserProfile,
                    persist=True,
                    resilient=True,
                    diagnostics=True,
                )
                joined = indexed_left_join(
                    left,
                    right,
                    join_key="id",
                    join_schema_map={"id": "Int64"},
                    persist=True,
                    resilient=True,
                    diagnostics=True,
                )
                result = joined.head()
```
