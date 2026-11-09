# Parquet

Parquet datasets as sources, as the target of `ParquetPipeline`, and composed with a live SQL table through `HybridDataset`.

## Parquet sources

```python
from boti_data import DataHelper, ParquetDataConfig

config = ParquetDataConfig(
    parquet_storage_path="/data/orders/",
    parquet_start_date=date(2024, 1, 1),
    parquet_end_date=date(2024, 3, 31),
)

with DataHelper(config) as helper:
    df = helper.pandas.load()
    df = helper.polars.load()
```

---

## Date-bounded Parquet source

```python
from boti_data import DataHelper, ParquetDataConfig
from datetime import date

config = ParquetDataConfig(
    parquet_storage_path="/data/warehouse/orders/",
    parquet_start_date=date(2024, 1, 1),
    parquet_end_date=date(2024, 6, 30),
)

with DataHelper(config) as helper:
    df = helper.pandas.load()
    ddf = helper.dask.load()  # lazy, partitioned read
```

## ParquetPipeline

`ParquetPipeline` is the first orchestration layer built on top of `DataHelper` and `HybridDataset`.
It materialises a lazy source load into a parquet dataset directory and can optionally reload the
written dataset through `ParquetReader`.

```python
from boti_data import DataHelper, ParquetPipeline

helper = DataHelper(
    backend="sqlalchemy",
    connection_url=f"sqlite:///{db_path}",
    poolclass="sqlalchemy.pool.NullPool",
    query_only=False,
    table="source_events",
)

pipeline = ParquetPipeline(
    helper,
    {
        "backend": "parquet",
        "storage_path": "/tmp/source_events_dataset",
        "partition_on": ["partition_date"],
    },
    date_field="event_date",
)

result = pipeline.materialize(
    filters={"status__exact": "active"},
    reload=True,
    reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
)

assert result.reloaded is True
frame = result.frame
```

Use `materialize()` / `amaterialize()` for the one-step workflow, or call `to_parquet()` and
`from_parquet()` separately when you want explicit control over the write and reload phases.
See `examples/data_parquet_pipeline.py` for a runnable end-to-end example.

## HybridDataset — historical + live composition

```python
historical = DataHelper(backend="parquet", storage_path="/data/historical", parquet_filename="orders")
live = DataHelper(backend="sqlalchemy", connection_url="sqlite:///live.db", table="orders_live")

dataset = HybridDataset(
    historical,
    live,
    date_field="order_date",
    split_date="2026-04-18",
)

# mixed window (auto -> dask for hybrid composition)
ddf = dataset.load(start="2026-04-15", end="2026-04-20", return_type="auto")

# explicit source override through engine-bound view
pdf = dataset.pandas.load(start="2026-04-18", end="2026-04-20", source="live")
```

## Connection catalog

```python
from boti_data import ConnectionCatalog, DataHelper

catalog = ConnectionCatalog()
catalog.load_sql("prod", prefix="PROD_DB_")  # reads PROD_DB_URL, PROD_DB_POOL_SIZE, etc.
catalog.load_sql("reporting", prefix="REPORT_DB_")

prod_config = catalog.sql_config("prod")
report_config = catalog.sql_config("reporting")

with DataHelper(prod_config, table="orders") as helper:
    df = helper.pandas.load(status="confirmed")
```

---

## Parquet in distributed settings

Parquet resources use `fsspec` for filesystem access. The filesystem object is not pickled directly; instead, `ParquetDataResource` uses a `fs_factory` callable or a `filesystem_profile` name that workers can use to reconstruct the filesystem independently. When `filesystem_profile` is used, filesystem creation is routed through `ConnectionCatalog` adapters so retry/cache behaviour stays consistent.

```python
from boti_data import DataHelper, ParquetDataConfig, ConnectionCatalog

catalog = ConnectionCatalog()
catalog.load_filesystem("s3_prod", prefix="S3_")  # reads S3_ENDPOINT, S3_KEY, etc.

config = ParquetDataConfig(
    filesystem_profile="s3_prod",  # workers resolve filesystem from catalog
    parquet_storage_path="s3://my-bucket/orders/",
)

with DataHelper.session(cluster_factory=LocalCluster) as client:
    with DataHelper(config) as helper:
        ddf = helper.dask.load()
        result = ddf.compute()
```

---
