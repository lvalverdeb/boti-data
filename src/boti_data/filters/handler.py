from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import dask.dataframe as dd
import pyarrow as pa
from boti.core.logger import Logger

from . import arrow_kernels
from .expressions import (
    And,
    ColOp,
    Expr,
    Not,
    Or,
    ParquetFilterGroup,
    ParquetFilters,
    TrueExpr,
)
from .parquet_compiler import ParquetFilterCompiler
from .utils import (
    COMPARISON_OPERATORS,
    DATE_OPERATORS,
    DT_OPERATORS,
    apply_operation_dask,
    apply_operation_sqlalchemy,
    get_backend_methods,
    get_dask_column,
    get_sqlalchemy_column,
    parse_filter_key,
    parse_filter_value,
    pushdown_ops,
    rewrite_date_logic,
    suggest_in_filter_chunking,
)


class FilterHandler:
    """Coordinates cross-backend filter parsing, pushdown, and expression execution."""

    COMPARISON_OPERATORS: ClassVar[list[str]] = COMPARISON_OPERATORS
    DT_OPERATORS: ClassVar[list[str]] = DT_OPERATORS
    DATE_OPERATORS: ClassVar[list[str]] = DATE_OPERATORS
    PARQUET_OP_MAP: ClassVar[dict[str, str]] = ParquetFilterCompiler.OPERATOR_MAP

    _parse_filter_key = staticmethod(parse_filter_key)
    _parse_filter_value = staticmethod(parse_filter_value)
    _get_backend_methods = staticmethod(get_backend_methods)
    _get_sqlalchemy_column = staticmethod(get_sqlalchemy_column)
    _get_dask_column = staticmethod(get_dask_column)
    _apply_operation_sqlalchemy = staticmethod(apply_operation_sqlalchemy)
    _apply_operation_dask = staticmethod(apply_operation_dask)
    _rewrite_date_logic = staticmethod(rewrite_date_logic)
    _suggest_in_filter_chunking = staticmethod(suggest_in_filter_chunking)

    def __init__(self, backend: str, logger: Any = None, debug: bool = False):
        self.backend = backend
        self.logger = logger or Logger.default_logger(logger_name=self.__class__.__name__)
        if hasattr(self.logger, "set_level"):
            self.logger.set_level(Logger.DEBUG if debug else Logger.INFO)
        self.backend_methods = self._get_backend_methods(backend)

    def _pushdown_ops(self) -> set[str]:
        base_ops = pushdown_ops()
        if self.backend == "arrow":
            return base_ops | arrow_kernels.ARROW_PUSHDOWN_OPS
        return base_ops

    def split_pushdown_and_residual(
        self,
        filters: dict[str, Any],
    ) -> tuple[ParquetFilters, dict[str, Any]]:
        # For Arrow backend, we allow $or/$not to be handled via compute kernels
        if self.backend != "arrow" and any(str(key).startswith("$") for key in filters):
            return [], filters

        pushdown_candidates: dict[str, Any] = {}
        residual_filters: dict[str, Any] = {}

        for key, value in filters.items():
            self._classify_pushdown_filter(
                key,
                value,
                pushdown_candidates=pushdown_candidates,
                residual_filters=residual_filters,
            )

        parquet_filters = self.to_parquet_filters(pushdown_candidates)
        return parquet_filters, residual_filters

    def _classify_pushdown_filter(
        self,
        key: str,
        value: Any,
        *,
        pushdown_candidates: dict[str, Any],
        residual_filters: dict[str, Any],
    ) -> None:
        if str(key).startswith("$"):
            residual_filters[key] = value
            return

        field, casting, op = self._parse_filter_key(key)
        if self._append_date_pushdown(field, casting, op, value, pushdown_candidates):
            return

        if casting is not None:
            residual_filters[key] = value
            return

        if op in self._pushdown_ops():
            pushdown_candidates[key] = value
            if self.backend == "dask" and op in {"in", "not_in"}:
                residual_filters[key] = value
            return

        residual_filters[key] = value

    def _append_date_pushdown(
        self,
        field: str,
        casting: str | None,
        op: str,
        value: Any,
        output: dict[str, Any],
    ) -> bool:
        if casting != "date" or op not in {"gt", "gte", "lt", "lte", "range"}:
            return False
        try:
            rewritten = self._rewrite_date_logic(op, value)
        except (ValueError, TypeError):
            return False
        for new_op, new_value in rewritten.items():
            output[f"{field}__{new_op}"] = new_value
        return True

    def to_parquet_filters(
        self,
        filters: dict[str, Any] | None = None,
    ) -> ParquetFilterGroup:
        return self._parquet_filter_compiler().compile(filters)

    def _append_parquet_filter(
        self,
        output: list[tuple[str, str, Any]],
        field: str,
        op: str,
        value: Any,
    ) -> None:
        output.extend(ParquetFilterCompiler.compile_parsed(field, op, value))

    def _parquet_filter_compiler(self) -> ParquetFilterCompiler:
        return ParquetFilterCompiler(
            parse_key=self._parse_filter_key,
            parse_value=self._parse_filter_value,
            pushdown_ops=self._pushdown_ops,
        )

    @staticmethod
    def _combine_and(exprs: list[Expr]) -> Expr:
        filtered_exprs = [expr for expr in exprs if not isinstance(expr, TrueExpr)]
        if not filtered_exprs:
            return TrueExpr()
        expr = filtered_exprs[0]
        for subexpr in filtered_exprs[1:]:
            expr = And(expr, subexpr)
        return expr

    @staticmethod
    def _combine_or(exprs: list[Expr]) -> Expr:
        filtered_exprs = [expr for expr in exprs if not isinstance(expr, TrueExpr)]
        if not filtered_exprs:
            return TrueExpr()
        expr = filtered_exprs[0]
        for subexpr in filtered_exprs[1:]:
            expr = Or(expr, subexpr)
        return expr

    def compile_filters(self, filters: dict[str, Any] | None = None) -> Expr:
        filters = filters or {}
        if not filters:
            return TrueExpr()

        clauses: list[Expr] = []

        if "$and" in filters:
            clauses.extend(self.compile_filters(sub) for sub in filters["$and"])

        if "$or" in filters:
            clauses.append(self._combine_or([self.compile_filters(sub) for sub in filters["$or"]]))

        if "$not" in filters:
            clauses.append(Not(self.compile_filters(filters["$not"])))

        for key, value in filters.items():
            if key.startswith("$"):
                continue
            field, casting, op = self._parse_filter_key(key)
            clauses.append(
                ColOp(
                    field=field,
                    casting=casting,
                    op=op,
                    value=value,
                    handler=self,
                )
            )

        return self._combine_and(clauses)

    def build_mask_fn(
        self,
        filters: dict[str, Any] | None = None,
    ) -> Callable[[dd.DataFrame], dd.Series]:
        expr = self.compile_filters(filters)

        def _mask(df: dd.DataFrame) -> dd.Series:
            return expr.mask(df)

        return _mask

    def build_sqlalchemy_condition(
        self,
        model: Any,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        expr = self.compile_filters(filters)
        return expr.to_sqlalchemy_condition(model)

    def suggest_sql_in_chunking(
        self,
        filters: dict[str, Any] | None = None,
        *,
        chunk_size: int = 900,
        max_concurrency: int = 8,
    ) -> dict[str, Any] | None:
        return self._suggest_in_filter_chunking(
            filters or {},
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
        )

    def apply_filters(self, query_or_df: Any, model: Any = None, filters: Any = None) -> Any:
        filters = filters or {}
        if self.backend == "sqlalchemy":
            if model is None:
                raise ValueError("model is required when applying SQLAlchemy filters.")
            condition = self.build_sqlalchemy_condition(model, filters)
            if hasattr(query_or_df, "where"):
                return query_or_df.where(condition)
            if hasattr(query_or_df, "filter"):
                return query_or_df.filter(condition)
            raise TypeError("SQLAlchemy filter target must support .where(...) or .filter(...).")

        if self.backend == "dask":
            expr = self.compile_filters(filters)
            if expr.is_trivial():
                return query_or_df
            condition = expr.mask(query_or_df)
            return self.backend_methods["apply_condition"](query_or_df, condition)

        if self.backend == "arrow":
            if not isinstance(query_or_df, pa.Table):
                raise TypeError(
                    f"Arrow backend requires pyarrow.Table, got {type(query_or_df).__name__}."
                )
            return arrow_kernels.apply_arrow_filters(query_or_df, filters)

        raise ValueError(f"Unsupported backend: {self.backend}")
