from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .expressions import ParquetFilterGroup

ParseKey = Callable[[str], tuple[str, str | None, str]]
ParseValue = Callable[[str | None, Any], Any]
PushdownOps = Callable[[], set[str]]


class ParquetFilterCompiler:
    """Compiles high-level filter kwargs into PyArrow/Dask parquet pushdown filters."""

    OPERATOR_MAP: ClassVar[dict[str, str]] = {
        "exact": "=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "not_exact": "!=",
        "in": "in",
        "not_in": "not in",
    }

    def __init__(
        self,
        *,
        parse_key: ParseKey,
        parse_value: ParseValue,
        pushdown_ops: PushdownOps,
    ) -> None:
        self._parse_key = parse_key
        self._parse_value = parse_value
        self._pushdown_ops = pushdown_ops

    def compile(self, filters: dict[str, Any] | None = None) -> ParquetFilterGroup:
        output: ParquetFilterGroup = []
        for key, value in (filters or {}).items():
            compiled = self._compile_item(key, value)
            if compiled:
                output.extend(compiled)
        return output

    def _compile_item(self, key: str, value: Any) -> ParquetFilterGroup:
        if str(key).startswith("$"):
            return []

        field, casting, op = self._parse_key(key)
        if casting is not None or op not in self._pushdown_ops():
            return []

        parsed_value = self._parse_value(casting, value)
        return self.compile_parsed(field, op, parsed_value)

    @classmethod
    def compile_parsed(cls, field: str, op: str, value: Any) -> ParquetFilterGroup:
        if op == "range":
            return cls._compile_range(field, value)
        return cls._compile_comparison(field, op, value)

    @staticmethod
    def _compile_range(field: str, value: Any) -> ParquetFilterGroup:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return []
        lower, upper = value
        return [(field, ">=", lower), (field, "<=", upper)]

    @classmethod
    def _compile_comparison(cls, field: str, op: str, value: Any) -> ParquetFilterGroup:
        parquet_op = cls.OPERATOR_MAP.get(op)
        if parquet_op is None:
            return []

        if op in {"in", "not_in"}:
            value = list(value) if not isinstance(value, (list, tuple)) else value
            if not value:
                return []

        return [(field, parquet_op, value)]
