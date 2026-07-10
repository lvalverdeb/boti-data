"""Built-in DataFrame transformers for Silver-layer data processing.

Each transformer implements :class:`DataFrameTransformer` and performs
a single, focused transformation on a ``pd.DataFrame``.  Compose them
via :class:`CompositeTransformer` for multi-step pipelines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class TypeCaster:
    """Cast columns to specified pandas dtype strings.

    Parameters
    ----------
    casts:
        Mapping of ``{column_name: target_dtype}``.  Supported dtypes
        include anything accepted by ``pd.DataFrame.astype()``:
        ``"int64"``, ``"float64"``, ``"bool"``, ``"string"``, etc.

    Missing columns are silently skipped.
    """

    def __init__(self, casts: dict[str, str]) -> None:
        self._casts = dict(casts)

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        applicable = {col: dt for col, dt in self._casts.items() if col in df.columns}
        if applicable:
            df = df.astype(applicable, errors="ignore")
        return df


class RowFilter:
    """Filter rows using pandas query strings.

    Parameters
    ----------
    predicates:
        List of pandas query expressions (e.g. ``["active == True", "value > 0"]``).
        All predicates are applied in sequence (AND semantics).
    """

    def __init__(self, predicates: list[str]) -> None:
        self._predicates = list(predicates)

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        for expr in self._predicates:
            mask = df.eval(expr)
            df = df.loc[mask].reset_index(drop=True)
        return df


class DerivedColumn:
    """Add computed columns via :meth:`pd.DataFrame.eval`.

    Parameters
    ----------
    columns:
        Mapping of ``{new_column_name: eval_expression}``.  Expressions
        are evaluated in the DataFrame's namespace, so all existing
        columns are available as variables.

    Example::

        DerivedColumn({"sla_end": "arrival_date + pd.Timedelta(days=7)"})
    """

    def __init__(self, columns: dict[str, str]) -> None:
        self._columns = dict(columns)

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        for name, expr in self._columns.items():
            df[name] = df.eval(expr)
        return df


class Deduplicator:
    """Drop duplicate rows.

    Parameters
    ----------
    subset:
        Column names to consider for duplicate detection.  ``None``
        means all columns.
    keep:
        Which duplicate to keep: ``"first"``, ``"last"``, or ``False``
        (drop all duplicates).
    """

    def __init__(
        self,
        subset: list[str] | None = None,
        keep: str = "last",
    ) -> None:
        self._subset = subset
        self._keep = keep

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=self._subset, keep=self._keep).reset_index(
            drop=True,
        )
        removed = before - len(df)
        if removed:
            import logging
            logging.getLogger(__name__).debug(
                "Deduplicator removed %d duplicate rows", removed,
            )
        return df
