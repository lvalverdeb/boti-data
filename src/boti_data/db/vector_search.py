"""
Nearest-neighbour query helpers for pgvector columns.

Builds an ``ORDER BY <distance> LIMIT k`` SQLAlchemy ``Select`` for a vector
column, with the query vector bound as a driver parameter (via SQLAlchemy's
own ``ColumnOperators.op()``) rather than interpolated into the SQL string.
Deliberately duck-typed against the column's comparator instead of importing
pgvector directly: this module has no import-time dependency on the
``pgvector`` package, so it stays available on the base install. Using it on
a non-vector column (or a vector column reflected without ``boti-data[pgvector]``
installed, see ``boti_data.db.sql_model_registry``) raises a clear ``TypeError``
rather than silently building broken SQL.

Pass the returned statement straight to ``DataGateway.load(statement=...,
model=...)`` or a plain SQLAlchemy session.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy import Select, select

VectorMetric = Literal["cosine", "l2", "inner_product", "l1"]

# Mirrors pgvector.sqlalchemy.Vector.Comparator's method names.
_METRIC_METHODS: dict[str, str] = {
    "cosine": "cosine_distance",
    "l2": "l2_distance",
    "inner_product": "max_inner_product",
    "l1": "l1_distance",
}


def vector_distance(
    column: Any, query_vector: Sequence[float], *, metric: VectorMetric = "cosine"
) -> Any:
    """Build a distance expression for ``column`` against ``query_vector``.

    ``query_vector`` is bound as a parameter by the underlying pgvector
    comparator method, never string-interpolated.
    """
    method_name = _METRIC_METHODS.get(metric)
    if method_name is None:
        raise ValueError(f"Unsupported metric {metric!r}; choose one of {sorted(_METRIC_METHODS)}.")
    method = getattr(column, method_name, None)
    if method is None:
        raise TypeError(
            f"Column {column!r} does not support vector distance operators. Is it a "
            "pgvector.sqlalchemy.Vector column, and is boti-data[pgvector] installed?"
        )
    return method(list(query_vector))


def nearest_neighbors(
    model: Any,
    column: Any,
    query_vector: Sequence[float],
    *,
    k: int,
    metric: VectorMetric = "cosine",
) -> Select:
    """``SELECT * FROM model ORDER BY column <metric> query_vector LIMIT k``.

    ``column`` is the mapped attribute to search against (e.g. ``Model.vector``).
    Add further filtering with ``.where(...)`` on the returned statement before
    passing it to ``DataGateway.load(statement=..., model=model)``.
    """
    distance = vector_distance(column, query_vector, metric=metric)
    return select(model).order_by(distance).limit(k)
