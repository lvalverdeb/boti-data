from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

import boti_data.joins as joins_module
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.schema import validate_schema


def _build_left_right_dask_frames() -> tuple[dd.DataFrame, dd.DataFrame]:
    left = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3], dtype="Int64"),
                "left_value": ["a", "b", "c"],
            }
        ),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series([1, 3], dtype="Int64"),
                "right_value": ["x", "z"],
            }
        ),
        npartitions=2,
    )
    return left, right


def test_left_join_frames_normalizes_join_keys_for_pandas() -> None:
    left = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="Int64"),
            "left_value": ["a", "b", "c"],
        }
    )
    right = pd.DataFrame(
        {
            "id": pd.Series(["1", "3"], dtype="string"),
            "right_value": ["x", "z"],
        }
    )

    joined = left_join_frames(left, right, left_on=["id"], join_schema_map={"id": "Int64"})

    validate_schema(joined, {"id": "Int64"}, require_columns=True)
    assert joined.loc[0, "right_value"] == "x"
    assert pd.isna(joined.loc[1, "right_value"])
    assert joined.loc[2, "right_value"] == "z"


def test_left_join_frames_normalizes_join_keys_for_dask() -> None:
    left = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3], dtype="Int64"),
                "left_value": ["a", "b", "c"],
            }
        ),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series(["1", "3"], dtype="string"),
                "right_value": ["x", "z"],
            }
        ),
        npartitions=1,
    )

    joined = left_join_frames(left, right, left_on=["id"], join_schema_map={"id": "Int64"})
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "right_value"] == "x"
    assert pd.isna(computed.loc[1, "right_value"])
    assert computed.loc[2, "right_value"] == "z"


def test_indexed_left_join_returns_expected_matches_for_dask() -> None:
    left = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3, 4], dtype="Int64"),
                "left_value": ["a", "b", "c", "d"],
            }
        ),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series(["1", "2", "3"], dtype="string"),
                "right_value": ["x", "y", "z"],
            }
        ),
        npartitions=2,
    )

    joined = indexed_left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=True,
    )
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    validate_schema(computed, {"id": "Int64"}, require_columns=True)
    assert computed.loc[0, "right_value"] == "x"
    assert computed.loc[1, "right_value"] == "y"
    assert computed.loc[2, "right_value"] == "z"
    assert pd.isna(computed.loc[3, "right_value"])


def test_indexed_left_join_computes_with_distributed_client() -> None:
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    left = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series(range(1, 1001), dtype="Int64"),
                "left_value": [f"l-{idx}" for idx in range(1, 1001)],
            }
        ),
        npartitions=4,
    )
    right = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series(range(1, 751), dtype="Int64"),
                "right_value": [f"r-{idx}" for idx in range(1, 751)],
            }
        ),
        npartitions=3,
    )

    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as cluster,
        Client(cluster),
    ):
        joined = indexed_left_join(
            left,
            right,
            join_key="id",
            join_schema_map={"id": "Int64"},
            persist=True,
        )
        computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed["right_value"].count() == 750
    assert computed["right_value"].isna().sum() == 250


def test_align_join_columns_skips_noop_pandas_realignment(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="Int64"),
            "value": ["a", "b", "c"],
        }
    )
    monkeypatch.setattr(
        joins_module,
        "apply_schema_map",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("apply_schema_map should not run")
        ),
    )

    result = joins_module._align_join_columns(frame, {"id": "Int64"})

    assert result is frame


def test_align_join_columns_skips_noop_dask_realignment(monkeypatch) -> None:
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3], dtype="Int64"),
                "value": ["a", "b", "c"],
            }
        ),
        npartitions=2,
    )
    monkeypatch.setattr(
        joins_module,
        "apply_schema_map",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("apply_schema_map should not run")
        ),
    )

    result = joins_module._align_join_columns(frame, {"id": "Int64"})

    assert result is frame


def test_indexed_left_join_uses_safe_persist_when_resilient(monkeypatch) -> None:
    left, right = _build_left_right_dask_frames()
    calls: list[int] = []

    def fake_safe_persist(frame, *, dask_client=None, logger=None) -> dd.DataFrame:
        calls.append(frame.npartitions)
        return frame.persist()

    monkeypatch.setattr(joins_module, "safe_persist", fake_safe_persist)

    joined = indexed_left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=True,
        resilient=True,
    )
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert len(calls) == 3
    assert computed.loc[0, "right_value"] == "x"
    assert pd.isna(computed.loc[1, "right_value"])
    assert computed.loc[2, "right_value"] == "z"


def test_indexed_left_join_dry_run_skips_persist_and_logs(monkeypatch) -> None:
    left, right = _build_left_right_dask_frames()

    logger = type(
        "Logger",
        (),
        {
            "infos": [],
            "warnings": [],
            "info": lambda self, message, *_args, **_kwargs: self.infos.append(message),
            "warning": lambda self, message, *_args, **_kwargs: self.warnings.append(message),
        },
    )()

    def fail_safe_persist(*_args, **_kwargs) -> None:
        raise AssertionError("safe_persist should not run in dry_run mode")

    monkeypatch.setattr(joins_module, "safe_persist", fail_safe_persist)

    joined = indexed_left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=True,
        resilient=True,
        diagnostics=True,
        dry_run=True,
        logger=logger,
    )
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "right_value"] == "x"
    assert any("dry run requested; persist steps skipped" in message for message in logger.infos)
    assert any(
        "dry run requested; execution skipped with graph=" in message for message in logger.infos
    )
