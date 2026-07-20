"""
Filter tests for the ``sqlalchemy`` backend: column resolution, LIKE/regex
operators, and IN/NOT IN null-handling semantics.

Split out of test_filters.py purely for god-module headroom.
"""

import pytest
from sqlalchemy import ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from boti_data.filters import FilterHandler


@pytest.fixture
def sql_handler() -> FilterHandler:
    return FilterHandler(backend="sqlalchemy", debug=True)


def test_apply_filters_supports_sqlalchemy_select(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))
        description: Mapped[str] = mapped_column(String(128))

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                User(status="active", description="urgent order"),
                User(status="inactive", description="backlog item"),
            ]
        )
        session.commit()

        stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"status__exact": "active", "description__icontains": "urgent"},
        )
        rows = session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].status == "active"


def test_get_sqlalchemy_column_rejects_relationships(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class Team(Base):
        __tablename__ = "teams"

        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(32))

    class User(Base):
        __tablename__ = "users_with_team"

        id: Mapped[int] = mapped_column(primary_key=True)
        team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
        team: Mapped[Team] = relationship()

    with pytest.raises(AttributeError, match="directly mapped column"):
        sql_handler._get_sqlalchemy_column("team", User, None)


def test_sqlalchemy_like_operations_escape_wildcards(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "escaped_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        description: Mapped[str] = mapped_column(String(128))

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                User(description="promo 50%_OFF"),
                User(description="promo 50AAOFF"),
            ]
        )
        session.commit()

        contains_stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"description__contains": "50%_OFF"},
        )
        contains_rows = session.execute(contains_stmt).scalars().all()

        iexact_stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"description__iexact": "PROMO 50%_OFF"},
        )
        iexact_rows = session.execute(iexact_stmt).scalars().all()

    assert [row.description for row in contains_rows] == ["promo 50%_OFF"]
    assert [row.description for row in iexact_rows] == ["promo 50%_OFF"]


def test_sqlalchemy_regex_uses_dialect_agnostic_operator(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "regex_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        description: Mapped[str] = mapped_column(String(128))

    stmt = sql_handler.apply_filters(
        select(User),
        model=User,
        filters={"description__regex": "^urgent"},
    )
    compiled = str(stmt.compile(create_engine("sqlite:///:memory:")))

    assert "~" not in compiled
    assert "REGEXP" in compiled.upper()


def test_sqlalchemy_in_filter_normalizes_duplicates_and_nulls(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "nullable_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        code: Mapped[str | None] = mapped_column(String(32), nullable=True)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                User(code="A"),
                User(code="B"),
                User(code=None),
            ]
        )
        session.commit()

        stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"code__in": ["A", "A", None]},
        )
        rows = session.execute(stmt).scalars().all()

    assert [row.code for row in rows] == ["A", None]


def test_sqlalchemy_not_in_filter_handles_empty_and_nulls(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "nullable_users_not_in"

        id: Mapped[int] = mapped_column(primary_key=True)
        code: Mapped[str | None] = mapped_column(String(32), nullable=True)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                User(code="A"),
                User(code="B"),
                User(code=None),
            ]
        )
        session.commit()

        empty_stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"code__not__in": []},
        )
        null_stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"code__not__in": [None, "A", "A"]},
        )
        empty_rows = session.execute(empty_stmt).scalars().all()
        null_rows = session.execute(null_stmt).scalars().all()

    assert [row.code for row in empty_rows] == ["A", "B", None]
    assert [row.code for row in null_rows] == ["B"]


def test_sqlalchemy_empty_in_filter_returns_no_rows(sql_handler) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "empty_in_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        code: Mapped[str] = mapped_column(String(32))

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all([User(code="A"), User(code="B")])
        session.commit()

        stmt = sql_handler.apply_filters(
            select(User),
            model=User,
            filters={"code__in": []},
        )
        rows = session.execute(stmt).scalars().all()

    assert rows == []


def test_filter_handler_suggests_sql_in_chunking(sql_handler) -> None:
    suggestion = sql_handler.suggest_sql_in_chunking(
        {"code__in": [1, 2, 2, None, 3, 4, 5]},
        chunk_size=2,
        max_concurrency=4,
    )

    assert suggestion == {
        "filter_key": "code__in",
        "value_count": 5,
        "in_chunk_size": 2,
        "chunk_count": 3,
        "in_chunk_concurrency": 3,
    }
