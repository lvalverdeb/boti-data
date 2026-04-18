from __future__ import annotations

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
from sqlalchemy import Date, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataGateway, DataHelper, ParquetReader
from boti_data.db import SqlDatabaseConfig
from boti_data.parquet import ParquetDataConfig


class LegacyBase(DeclarativeBase):
    pass


class LegacyProduct(LegacyBase):
    __tablename__ = "legacy_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_tipo_produto: Mapped[int] = mapped_column()
    codigo_barra: Mapped[str] = mapped_column(String(32))


def test_helper_from_legacy_config_loads_sql_configured_mode(tmp_path):
    db_path = tmp_path / "helper_legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        LegacyBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    LegacyProduct(id_tipo_produto=1, codigo_barra="A"),
                    LegacyProduct(id_tipo_produto=2, codigo_barra="B"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper.from_legacy_config(
        {
            "backend": "sqlalchemy",
            "connection_url": f"sqlite:///{db_path}",
            "poolclass": "sqlalchemy.pool.NullPool",
            "query_only": False,
            "table": "legacy_products",
            "field_map": {"id_tipo_produto": "product_type_id", "codigo_barra": "barcode"},
            "sticky_filters": {"product_type_id": 1},
            "df_params": {"fieldnames": ("product_type_id", "barcode")},
        }
    )
    try:
        frame = helper.load(as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["product_type_id"].tolist() == [1]
    assert frame["barcode"].tolist() == ["A"]


def test_helper_kwargs_init_loads_sql_configured_mode(tmp_path):
    db_path = tmp_path / "helper_kwargs_legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        LegacyBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    LegacyProduct(id_tipo_produto=1, codigo_barra="A"),
                    LegacyProduct(id_tipo_produto=2, codigo_barra="B"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="legacy_products",
        field_map={"id_tipo_produto": "product_type_id", "codigo_barra": "barcode"},
        sticky_filters={"product_type_id": 1},
        df_params={"fieldnames": ("product_type_id", "barcode")},
    )
    try:
        frame = helper.load(as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["product_type_id"].tolist() == [1]
    assert frame["barcode"].tolist() == ["A"]


def test_helper_can_wrap_existing_gateway(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "wrapped_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "wrapped_helper.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(User(status="active"))
            session.commit()
    finally:
        engine.dispose()

    gateway = DataGateway(
        SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )
    )
    helper = DataHelper.from_gateway(gateway)
    try:
        frame = helper.load(statement=select(User), model=User, as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_helper_dask_view_defaults_to_lazy_dask(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users",
    )
    try:
        frame = helper.dask.load(status__exact="active")
    finally:
        helper.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active"]


def test_helper_pandas_view_defaults_to_eager_pandas(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users_pandas"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper_pandas.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users_pandas",
    )
    try:
        frame = helper.pandas.load(status__exact="active")
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_helper_polars_view_defaults_to_eager_polars(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users_polars"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper_polars.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users_polars",
    )
    try:
        frame = helper.polars.load(status__exact="active")
    finally:
        helper.close()

    assert isinstance(frame, pl.DataFrame)
    assert frame["status"].to_list() == ["active"]


def test_helper_dask_view_rejects_incompatible_overrides(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users_dask_strict"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper_dask_strict.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users_dask_strict",
    )
    try:
        with pytest.raises(ValueError, match="helper.dask does not allow return_type='polars'"):
            helper.dask.load(return_type="polars")
        with pytest.raises(ValueError, match="helper.dask does not allow as_pandas=True"):
            helper.dask.load(as_pandas=True)
    finally:
        helper.close()


def test_helper_pandas_view_rejects_incompatible_execution_mode(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users_pandas_strict"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper_pandas_strict.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users_pandas_strict",
    )
    try:
        with pytest.raises(ValueError, match="helper.pandas does not allow execution_mode='lazy'"):
            helper.pandas.load(execution_mode="lazy")
    finally:
        helper.close()


def test_helper_polars_view_rejects_incompatible_execution_mode(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "engine_view_users_polars_strict"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "engine_view_helper_polars_strict.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="engine_view_users_polars_strict",
    )
    try:
        with pytest.raises(ValueError, match="helper.polars does not allow execution_mode='lazy'"):
            helper.polars.load(execution_mode="lazy")
    finally:
        helper.close()


def test_helper_from_legacy_config_loads_parquet_with_legacy_storage_path_and_pyarrow_fs(temp_project_root):
    file_path = temp_project_root / "legacy_parquet" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    helper = DataHelper.from_legacy_config(
        {
            "backend": "parquet",
            "fs": pafs.LocalFileSystem(),
            "storage_path": str(file_path.parent),
            "parquet_filename": "users",
        }
    )
    try:
        frame = helper.load(filters={"status__exact": "active"}, as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1]


def test_helper_kwargs_init_loads_parquet_with_legacy_storage_path_and_pyarrow_fs(temp_project_root):
    file_path = temp_project_root / "legacy_kwargs_parquet" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    helper = DataHelper(
        backend="parquet",
        fs=pafs.LocalFileSystem(),
        storage_path=str(file_path.parent),
        parquet_filename="users",
    )
    try:
        frame = helper.load(filters={"status__exact": "active"}, as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1]


def test_helper_parquet_polars_view_defaults_to_eager_polars(temp_project_root):
    file_path = temp_project_root / "helper_engine_views" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    helper = DataHelper(
        backend="parquet",
        storage_path=str(file_path.parent),
        parquet_filename="users",
    )
    try:
        frame = helper.polars.load(filters={"status__exact": "active"})
    finally:
        helper.close()

    assert isinstance(frame, pl.DataFrame)
    assert frame["id"].to_list() == [1]


def test_helper_from_legacy_config_defaults_to_partition_date_hive_pruning(temp_project_root):
    root = temp_project_root / "legacy_hive_partitioned"
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-04-01").mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["in_range"]}).to_parquet(
        root / "partition_date=2026-01-01" / "part1.parquet",
        index=False,
    )
    pd.DataFrame({"id": [2], "status": ["out_of_range"]}).to_parquet(
        root / "partition_date=2026-04-01" / "part2.parquet",
        index=False,
    )

    helper = DataHelper.from_legacy_config(
        {
            "backend": "parquet",
            "storage_path": str(root),
            "parquet_start_date": "2026-01-01",
            "parquet_end_date": "2026-03-31",
        }
    )
    try:
        frame = helper.load(as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1]
    assert frame["status"].tolist() == ["in_range"]


def test_parquet_reader_defaults_to_partition_date_and_configured_eager_pandas(temp_project_root):
    file_path = temp_project_root / "reader_data" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        return_type="pandas",
        execution_mode="eager",
    )
    try:
        frame = reader.load()
    finally:
        reader.close()

    assert reader.partition_on == ["partition_date"]
    assert reader.parquet_storage_path == str(file_path.parent)
    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 2]


def test_parquet_reader_runtime_return_type_override_supports_lazy_dask(temp_project_root):
    file_path = temp_project_root / "reader_lazy" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(return_type="dask", execution_mode="lazy")
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


def test_parquet_reader_load_applies_filters_in_lazy_mode(temp_project_root):
    file_path = temp_project_root / "reader_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(filters={"status__exact": "active"})
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 3]


def test_parquet_reader_load_applies_in_filters_in_lazy_mode(temp_project_root):
    file_path = temp_project_root / "reader_in_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(filters={"associate_id__in": [2819, 4499]})
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


def test_parquet_reader_load_supports_legacy_bare_filter_kwargs(temp_project_root):
    file_path = temp_project_root / "reader_bare_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(associate_id__in=[2819, 4499])
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_eager_arrow_return_type(temp_project_root):
    file_path = temp_project_root / "reader_arrow" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = await reader.aload(return_type="arrow", execution_mode="eager")
    finally:
        await reader.aclose()

    assert isinstance(frame, pa.Table)
    assert frame.to_pydict()["status"] == ["active"]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_filters_with_explicit_pyarrow_fs(temp_project_root):
    file_path = temp_project_root / "reader_filter_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload(filters={"status__exact": "active"})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_in_filters_with_explicit_pyarrow_fs(temp_project_root):
    file_path = temp_project_root / "reader_in_filter_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload(filters={"associate_id__in": [2819, 4499]})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_legacy_bare_filter_kwargs(temp_project_root):
    file_path = temp_project_root / "reader_bare_async_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = await reader.aload(associate_id__in=[2819, 4499])
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_lazy_dask_with_explicit_pyarrow_fs(temp_project_root):
    file_path = temp_project_root / "reader_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload()
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 2]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_partition_filters_for_hive_layout(temp_project_root):
    root = temp_project_root / "reader_partition_filters"
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-01-02").mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(
        root / "partition_date=2026-01-01" / "part1.parquet",
        index=False,
    )
    pd.DataFrame({"id": [2], "status": ["inactive"]}).to_parquet(
        root / "partition_date=2026-01-02" / "part2.parquet",
        index=False,
    )

    reader = ParquetReader(storage_path=str(root))
    try:
        frame = await reader.aload(filters={"partition_date__exact": "2026-01-01"})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    computed = frame.compute().sort_values("id").reset_index(drop=True)
    assert computed["id"].tolist() == [1]
    assert computed["partition_date"].tolist() == ["2026-01-01"]


@pytest.mark.asyncio
async def test_helper_aloads_parquet(temp_project_root):
    file_path = temp_project_root / "data" / "helper_users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    helper = DataHelper(
        ParquetDataConfig(
            project_root=temp_project_root,
            parquet_storage_path=str(file_path.parent),
            parquet_filename="helper_users",
        )
    )
    try:
        frame = await helper.aload(filters={"status__exact": "active"})
    finally:
        await helper.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


def test_helper_load_period_uses_gateway_period_loader(tmp_path):
    class Base(DeclarativeBase):
        pass

    class Event(Base):
        __tablename__ = "helper_events"

        id: Mapped[int] = mapped_column(primary_key=True)
        event_date: Mapped[dt.date] = mapped_column(Date())
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "helper_period.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    Event(event_date=dt.date(2026, 1, 1), status="active"),
                    Event(event_date=dt.date(2026, 1, 2), status="inactive"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        {
            "backend": "sqlalchemy",
            "connection_url": f"sqlite:///{db_path}",
            "poolclass": "sqlalchemy.pool.NullPool",
            "query_only": False,
            "table": "helper_events",
            "df_params": {"fieldnames": ("event_date", "status")},
        }
    )
    try:
        frame = helper.load_period("event_date", "2026-01-01", "2026-01-01", as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_helper_left_join_uses_indexed_join_for_join_key():
    left = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 2, 3], dtype="Int64"), "left_value": ["a", "b", "c"]}),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 3], dtype="Int64"), "right_value": ["x", "z"]}),
        npartitions=1,
    )

    joined = DataHelper.left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=True,
    )
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "right_value"] == "x"
    assert pd.isna(computed.loc[1, "right_value"])
    assert computed.loc[2, "right_value"] == "z"


def test_helper_left_join_uses_column_join_when_join_key_missing():
    left = pd.DataFrame({"id": pd.Series([1, 2], dtype="Int64"), "left_value": ["a", "b"]})
    right = pd.DataFrame({"id": pd.Series([1], dtype="Int64"), "right_value": ["x"]})

    joined = DataHelper.left_join(
        left,
        right,
        left_on=["id"],
        join_schema_map={"id": "Int64"},
    )

    assert joined["right_value"].tolist()[0] == "x"
    assert pd.isna(joined["right_value"].tolist()[1])


def test_helper_session_creates_managed_dask_session():
    pytest.importorskip("dask.distributed")

    with DataHelper.session(
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "processes": False,
            "dashboard_address": ":0",
        }
    ) as client:
        summary = client.scheduler_info()

    assert summary["workers"]
