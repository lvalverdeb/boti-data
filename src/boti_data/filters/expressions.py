from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import true

if TYPE_CHECKING:
    from .handler import FilterHandler

ParquetFilter = tuple[str, str, Any]
ParquetFilterGroup = list[ParquetFilter]
ParquetFilters = list[ParquetFilter | ParquetFilterGroup]


class Expr:
    def mask(self, df: dd.DataFrame) -> dd.Series:
        raise NotImplementedError

    def is_trivial(self) -> bool:
        return False

    def to_parquet_filters(self) -> ParquetFilters:
        return []

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        raise NotImplementedError

    def __and__(self, other: Expr) -> Expr:
        return And(self, other)

    def __or__(self, other: Expr) -> Expr:
        return Or(self, other)

    def __invert__(self) -> Expr:
        return Not(self)


@dataclass(frozen=True)
class TrueExpr(Expr):
    """Matches all rows; useful as a neutral starting point."""

    def is_trivial(self) -> bool:
        return True

    def mask(self, df: dd.DataFrame) -> dd.Series:
        return df.map_partitions(
            lambda partition: pd.Series(True, index=partition.index),
            meta=pd.Series(dtype=bool),
        )

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        return true()


_PARQUET_FILTER_BUILDERS: dict[str, Callable[[str, Any], ParquetFilterGroup]] = {
    "exact": lambda field, value: [(field, "=", value)],
    "not_exact": lambda field, value: [(field, "!=", value)],
    "gt": lambda field, value: [(field, ">", value)],
    "gte": lambda field, value: [(field, ">=", value)],
    "lt": lambda field, value: [(field, "<", value)],
    "lte": lambda field, value: [(field, "<=", value)],
    "in": lambda field, value: [
        (field, "in", list(value) if not isinstance(value, list) else value)
    ],
    "not_in": lambda field, value: [
        (field, "not in", list(value) if not isinstance(value, list) else value)
    ],
    "range": lambda field, value: [
        (field, ">=", value[0]),
        (field, "<=", value[1]),
    ],
}


@dataclass(frozen=True)
class ColOp(Expr):
    field: str
    casting: str | None
    op: str
    value: Any
    handler: FilterHandler

    def mask(self, df: dd.DataFrame) -> dd.Series:
        column = self.handler._get_dask_column(df, self.field, self.casting)
        value = self.handler._parse_filter_value(self.casting, self.value)
        return self.handler._apply_operation_dask(column, self.op, value)

    def to_parquet_filters(self) -> ParquetFilterGroup:
        if self.op not in self.handler._pushdown_ops():
            return []

        value = self.handler._parse_filter_value(self.casting, self.value)
        builder = _PARQUET_FILTER_BUILDERS.get(self.op)
        if builder is None:
            return []
        return builder(self.field, value)

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        column = self.handler._get_sqlalchemy_column(self.field, model, self.casting)
        value = self.handler._parse_filter_value(self.casting, self.value)
        return self.handler._apply_operation_sqlalchemy(column, self.op, value)


@dataclass(frozen=True)
class And(Expr):
    left: Expr
    right: Expr

    def mask(self, df: dd.DataFrame) -> dd.Series:
        return self.left.mask(df) & self.right.mask(df)

    def to_parquet_filters(self) -> ParquetFilters:
        return [*self.left.to_parquet_filters(), *self.right.to_parquet_filters()]

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        return self.left.to_sqlalchemy_condition(model) & self.right.to_sqlalchemy_condition(model)


@dataclass(frozen=True)
class Or(Expr):
    left: Expr
    right: Expr

    def mask(self, df: dd.DataFrame) -> dd.Series:
        return self.left.mask(df) | self.right.mask(df)

    def to_parquet_filters(self) -> ParquetFilters:
        left_filters, right_filters = (
            self.left.to_parquet_filters(),
            self.right.to_parquet_filters(),
        )
        if not left_filters or not right_filters:
            return []
        return [left_filters, right_filters]

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        return self.left.to_sqlalchemy_condition(model) | self.right.to_sqlalchemy_condition(model)


@dataclass(frozen=True)
class Not(Expr):
    inner: Expr

    def mask(self, df: dd.DataFrame) -> dd.Series:
        return ~self.inner.mask(df)

    def to_parquet_filters(self) -> ParquetFilters:
        return []

    def to_sqlalchemy_condition(self, model: Any) -> Any:
        return ~self.inner.to_sqlalchemy_condition(model)
