"""
Tests for DataGateway.describe()/adescribe() and the DataHelper delegates —
the wishlist #6 lightweight schema/row-count discovery path.
"""

from __future__ import annotations

import pytest

from boti_data import DataGateway, ParquetDataConfig, SqlDatabaseConfig, TableDescription

from ._pipelines_shared import _build_source_helper


class _StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def test_helper_describe_reports_schema_and_exact_row_count(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    try:
        result = helper.describe("source_events")
    finally:
        helper.close()

    assert isinstance(result, TableDescription)
    assert result.table == "source_events"
    assert set(result.columns) == {"id", "event_date", "status"}
    assert result.row_count == 3
    assert result.row_count_is_exact is True


def test_helper_describe_caps_row_count_when_over_limit(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    try:
        result = helper.describe("source_events", row_count_limit=2)
    finally:
        helper.close()

    assert result.row_count == 2
    assert result.row_count_is_exact is False


def test_helper_describe_uses_configured_table_by_default(temp_project_root) -> None:
    """`table` is already set at construction on this fixture's helper (configured mode)."""
    helper = _build_source_helper(temp_project_root)
    try:
        result = helper.describe()
    finally:
        helper.close()

    assert result.table == "source_events"


def test_gateway_describe_requires_a_table(tmp_path) -> None:
    db_path = tmp_path / "no_table_configured.db"
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with DataGateway(config) as gateway:
        with pytest.raises(ValueError, match="require table="):
            gateway.describe()


@pytest.mark.asyncio
async def test_helper_adescribe_matches_sync_describe(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    try:
        result = await helper.adescribe("source_events")
    finally:
        helper.close()

    assert result.row_count == 3
    assert result.row_count_is_exact is True


def test_gateway_describe_raises_for_unsupported_backend(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "describe_events.parquet"
    file_path.parent.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame({"id": [1, 2]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=_StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="describe_events",
    )

    with DataGateway(config) as gateway:
        with pytest.raises(NotImplementedError, match="does not support describe"):
            gateway.describe("describe_events")
