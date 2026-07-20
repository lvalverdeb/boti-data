"""
Shared fixtures and helpers for the test_field_map_gateway_* modules.

Split out of test_field_map_gateway.py (originally a single 1668-line module)
so the legacy/semantic SQLite table definitions, DSN fixtures, and DataGateway
constructor helpers are defined once and reused across the sibling test files
instead of being duplicated in each of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pydantic import SecretStr
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway import DataGateway

# ---------------------------------------------------------------------------
# Shared field map (DB legacy name → semantic name)
# ---------------------------------------------------------------------------

PRODUCT_MAP: dict[str, str] = {
    "id_tipo_produto": "product_type_id",
    "id_track_global": "global_track_id",
    "id_track_proceso": "process_track_id",
    "codigo_barra": "barcode",
    "nombre_th": "first_name",
}


# ===========================================================================
# DataGateway configured mode — SQLite in-memory databases
# ===========================================================================
#
# Two table structures are used:
#
#   LegacyProduct  — DB column names are the NON-semantic (legacy) names
#                    (id_tipo_produto, id_track_global, …)
#                    Used with field_map present.
#
#   SemanticProduct — DB column names ARE the semantic names
#                    (product_type_id, global_track_id, …)
#                    Used without a field_map.
# ---------------------------------------------------------------------------


class LegacyBase(DeclarativeBase):
    pass


class LegacyProduct(LegacyBase):
    """Table whose DB column names are the legacy (non-semantic) names."""

    __tablename__ = "legacy_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_tipo_produto: Mapped[int] = mapped_column()
    id_track_global: Mapped[int] = mapped_column()
    id_track_proceso: Mapped[int] = mapped_column()
    codigo_barra: Mapped[str] = mapped_column(String(32))
    nombre_th: Mapped[str] = mapped_column(String(64))


class SemanticBase(DeclarativeBase):
    pass


class SemanticProduct(SemanticBase):
    """Table whose DB column names are already the semantic names."""

    __tablename__ = "semantic_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_type_id: Mapped[int] = mapped_column()
    global_track_id: Mapped[int] = mapped_column()
    process_track_id: Mapped[int] = mapped_column()
    barcode: Mapped[str] = mapped_column(String(32))
    first_name: Mapped[str] = mapped_column(String(64))


ROWS = (
    # (type_id, global_track, process_track, barcode, name)
    (1, 10, 100, "A001", "Alice"),
    (1, 20, 200, "B002", "Bob"),
    (2, 30, 300, "C003", "Carol"),
)


@pytest.fixture(scope="module")
def legacy_dsn(tmp_path_factory) -> str:
    db_path = tmp_path_factory.mktemp("db") / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    LegacyBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                LegacyProduct(
                    id_tipo_produto=t,
                    id_track_global=g,
                    id_track_proceso=p,
                    codigo_barra=b,
                    nombre_th=n,
                )
                for t, g, p, b, n in ROWS
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def semantic_dsn(tmp_path_factory) -> str:
    db_path = tmp_path_factory.mktemp("db") / "semantic.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SemanticBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                SemanticProduct(
                    product_type_id=t,
                    global_track_id=g,
                    process_track_id=p,
                    barcode=b,
                    first_name=n,
                )
                for t, g, p, b, n in ROWS
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


def _legacy_gw(dsn, **kwargs) -> DataGateway:
    """DataGateway over the legacy-column DB (field_map present → translate)."""
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(
        config,
        table="legacy_products",
        field_map=PRODUCT_MAP,
        **kwargs,
    )


def _semantic_gw(dsn, **kwargs) -> DataGateway:
    """DataGateway over the semantic-column DB (no field_map → no translation)."""
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(
        config,
        table="semantic_products",
        **kwargs,
    )


_GLOBAL_TRACK_ID_EXTRACTORS: list[tuple[type, Callable[[Any], list]]] = [
    (dd.DataFrame, lambda frame: frame.compute()["global_track_id"].tolist()),
    (pd.DataFrame, lambda frame: frame["global_track_id"].tolist()),
    (pa.Table, lambda frame: frame["global_track_id"].to_pylist()),
    (pl.DataFrame, lambda frame: frame["global_track_id"].to_list()),
]


def _global_track_ids(frame):
    for frame_type, extract in _GLOBAL_TRACK_ID_EXTRACTORS:
        if isinstance(frame, frame_type):
            return extract(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")
