"""
Tests for boti_data.db.vector_search — nearest-neighbour query helpers for
pgvector columns.

Statement construction/compilation is fully hermetic (no live Postgres
needed); the full get-a-row-back-from-a-real-table path was verified during
development against a real Postgres+pgvector instance.
"""

import pgvector.sqlalchemy
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from boti_data.db.vector_search import nearest_neighbors, vector_distance


class _Base(DeclarativeBase):
    pass


class _Embedding(_Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    vector = mapped_column(pgvector.sqlalchemy.Vector(3))


def test_nearest_neighbors_binds_query_vector_as_a_parameter() -> None:
    stmt = nearest_neighbors(_Embedding, _Embedding.vector, [0.1, 0.2, 0.3], k=5)
    compiled = stmt.compile(dialect=postgresql.dialect())
    sql_text = str(compiled)

    assert "ORDER BY" in sql_text
    assert "LIMIT" in sql_text
    # The query vector must never be interpolated into the SQL text.
    assert "0.1" not in sql_text
    assert "0.2" not in sql_text
    assert compiled.params["vector_1"] == [0.1, 0.2, 0.3]
    assert compiled.params["param_1"] == 5


@pytest.mark.parametrize(
    ("metric", "operator"),
    [
        ("cosine", "<=>"),
        ("l2", "<->"),
        ("inner_product", "<#>"),
        ("l1", "<+>"),
    ],
)
def test_vector_distance_supports_every_pgvector_metric(metric: str, operator: str) -> None:
    distance = vector_distance(_Embedding.vector, [1.0, 2.0, 3.0], metric=metric)
    compiled = distance.compile(dialect=postgresql.dialect())
    assert operator in str(compiled)


def test_vector_distance_rejects_unsupported_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        vector_distance(_Embedding.vector, [1.0], metric="manhattan")


def test_vector_distance_rejects_non_vector_column() -> None:
    with pytest.raises(TypeError, match="does not support vector distance operators"):
        vector_distance(_Embedding.id, [1.0, 2.0])
