from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import dask.dataframe as dd

from boti_dask import (
    current_client_summary,
    describe_frame,
    diagnostics_logger,
    inspect_graph,
    safe_persist,
)
from boti_data.schema import (
    DataFrameLike,
    apply_schema_map,
    normalize_dtype_alias,
    normalize_schema_map,
    validate_schema,
)


def _align_join_columns(
    frame: DataFrameLike,
    join_schema_map: Mapping[str, str],
) -> DataFrameLike:
    normalized_schema = normalize_schema_map(join_schema_map)
    join_columns = list(normalized_schema)
    missing = [column for column in join_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing join column(s): {missing}")
    if all(
        normalize_dtype_alias(frame.dtypes[column]) == normalized_schema[column]
        for column in join_columns
    ):
        return frame
    aligned_subset = apply_schema_map(
        frame[join_columns],
        normalized_schema,
        require_columns=True,
    )
    validate_schema(aligned_subset, normalized_schema, require_columns=True)

    if isinstance(frame, dd.DataFrame):
        return frame.assign(**{column: aligned_subset[column] for column in join_columns})

    result = frame.copy()
    for column in join_columns:
        result[column] = aligned_subset[column]
    return result


def left_join_frames(
    left: DataFrameLike,
    right: DataFrameLike,
    *,
    left_on: Sequence[str],
    right_on: Sequence[str] | None = None,
    join_schema_map: Mapping[str, str],
    diagnostics: bool = False,
    dry_run: bool = False,
    logger: Any | None = None,
    label: str = "left_join_frames",
) -> DataFrameLike:
    """Left join two frames after normalizing join-key dtypes to an explicit schema."""
    active_logger = diagnostics_logger(logger, name="boti_data.joins") if diagnostics else None
    resolved_right_on = list(right_on or left_on)
    resolved_left_on = list(left_on)
    if len(resolved_left_on) != len(resolved_right_on):
        raise ValueError("left_on and right_on must have the same number of columns.")

    started = perf_counter()
    left_aligned = _align_join_columns(left, join_schema_map)
    right_aligned = _align_join_columns(right, join_schema_map)
    if diagnostics and isinstance(left_aligned, dd.DataFrame) and isinstance(right_aligned, dd.DataFrame):
        active_logger.warning(
            f"{label}: direct Dask merge may trigger a full shuffle for large inputs. "
            "Prefer indexed_left_join() for large distributed joins."
        )
        client_summary = current_client_summary()
        if client_summary is not None:
            active_logger.info(f"{label}: active client={client_summary}")

    joined = left_aligned.merge(
        right_aligned,
        how="left",
        left_on=resolved_left_on,
        right_on=resolved_right_on,
    )
    if dry_run and diagnostics and active_logger is not None and isinstance(joined, dd.DataFrame):
        active_logger.info(
            f"{label}: dry run requested; execution skipped with graph={inspect_graph(joined)}"
        )
    if diagnostics and active_logger is not None:
        active_logger.info(
            f"{label}: completed in {perf_counter() - started:.2f}s with metrics={describe_frame(joined)}"
        )
    return joined


def indexed_left_join(
    left: DataFrameLike,
    right: DataFrameLike,
    *,
    join_key: str,
    join_schema_map: Mapping[str, str],
    persist: bool = False,
    resilient: bool = False,
    reset_index: bool = True,
    diagnostics: bool = False,
    dry_run: bool = False,
    logger: Any | None = None,
    label: str = "indexed_left_join",
) -> DataFrameLike:
    """Align two frames on a join key, index them, and perform a left join.

    This is the preferred pattern for large Dask joins: pay the repartition/index
    cost once, persist both sides, then join on aligned indexes.
    """
    active_logger = diagnostics_logger(logger, name="boti_data.joins") if diagnostics else None
    started = perf_counter()
    left_aligned = _align_join_columns(left, join_schema_map)
    right_aligned = _align_join_columns(right, join_schema_map)
    if diagnostics and active_logger is not None:
        client_summary = current_client_summary()
        if client_summary is not None:
            active_logger.info(f"{label}: active client={client_summary}")
        active_logger.info(
            f"{label}: left={describe_frame(left_aligned)} right={describe_frame(right_aligned)} "
            f"join_key={join_key} persist={persist} resilient={resilient}"
        )
        if isinstance(left_aligned, dd.DataFrame):
            active_logger.info(f"{label}: left_graph={inspect_graph(left_aligned)}")
        if isinstance(right_aligned, dd.DataFrame):
            active_logger.info(f"{label}: right_graph={inspect_graph(right_aligned)}")
        if isinstance(left_aligned, dd.DataFrame) and not persist:
            active_logger.warning(
                f"{label}: persist=False may recompute expensive set_index/join stages on repeated downstream actions."
            )
    left_indexed = left_aligned.set_index(join_key)
    right_indexed = right_aligned.set_index(join_key)

    if dry_run and persist and diagnostics and active_logger is not None:
        active_logger.info(f"{label}: dry run requested; persist steps skipped.")
    if persist and not dry_run and isinstance(left_indexed, dd.DataFrame):
        left_indexed = (
            safe_persist(left_indexed, logger=active_logger)
            if resilient
            else left_indexed.persist()
        )
    if persist and not dry_run and isinstance(right_indexed, dd.DataFrame):
        right_indexed = (
            safe_persist(right_indexed, logger=active_logger)
            if resilient
            else right_indexed.persist()
        )

    joined = left_indexed.join(right_indexed, how="left")
    if persist and not dry_run and isinstance(joined, dd.DataFrame):
        joined = safe_persist(joined, logger=active_logger) if resilient else joined.persist()

    if reset_index:
        joined = joined.reset_index()
    if dry_run and diagnostics and active_logger is not None and isinstance(joined, dd.DataFrame):
        active_logger.info(
            f"{label}: dry run requested; execution skipped with graph={inspect_graph(joined)}"
        )
    if diagnostics and active_logger is not None:
        active_logger.info(
            f"{label}: completed in {perf_counter() - started:.2f}s with metrics={describe_frame(joined)}"
        )
    return joined


__all__ = ["indexed_left_join", "left_join_frames"]
