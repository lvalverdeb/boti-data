"""
DataHelper construction and engine-view tests: legacy config loading,
kwargs-based init, gateway wrapping, and the dask/pandas/polars engine
views' defaults and strict-mode rejections.

Split out of test_helper.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow.fs as pafs
import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataGateway, DataHelper
from boti_data.db import SqlDatabaseConfig


class LegacyBase(DeclarativeBase):
    pass


class LegacyProduct(LegacyBase):
    __tablename__ = "legacy_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_tipo_produto: Mapped[int] = mapped_column()
    codigo_barra: Mapped[str] = mapped_column(String(32))


def test_helper_from_legacy_config_loads_sql_configured_mode(tmp_path) -> None:
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


def test_helper_kwargs_init_loads_sql_configured_mode(tmp_path) -> None:
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


def test_helper_can_wrap_existing_gateway(tmp_path) -> None:
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


def test_helper_dask_view_defaults_to_lazy_dask(tmp_path) -> None:
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


def test_helper_pandas_view_defaults_to_eager_pandas(tmp_path) -> None:
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


def test_helper_polars_view_defaults_to_eager_polars(tmp_path) -> None:
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


def test_helper_dask_view_rejects_incompatible_overrides(tmp_path) -> None:
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


def test_helper_pandas_view_rejects_incompatible_execution_mode(tmp_path) -> None:
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


def test_helper_polars_view_rejects_incompatible_execution_mode(tmp_path) -> None:
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


def test_helper_from_legacy_config_loads_parquet_with_legacy_storage_path_and_pyarrow_fs(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "legacy_parquet" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

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


def test_helper_kwargs_init_loads_parquet_with_legacy_storage_path_and_pyarrow_fs(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "legacy_kwargs_parquet" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

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
