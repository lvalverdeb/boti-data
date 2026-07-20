"""
Filter tests for the ``dask`` backend: key parsing, pushdown/residual
splitting, AST compilation, and dask-column resolution.

Split out of test_filters.py purely for god-module headroom.
"""

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data.filters import And, ColOp, FilterHandler, Not, Or, TrueExpr


@pytest.fixture
def dask_handler() -> FilterHandler:
    return FilterHandler(backend="dask", debug=True)


def test_parse_filter_key(dask_handler) -> None:
    # Standard
    assert dask_handler._parse_filter_key("price__gt") == ("price", None, "gt")

    # With casting
    assert dask_handler._parse_filter_key("created_at__date__gte") == ("created_at", "date", "gte")

    # Negation alias mapping
    assert dask_handler._parse_filter_key("status__ne") == ("status", None, "not_exact")

    # Middle negation
    assert dask_handler._parse_filter_key("status__not__in") == ("status", None, "not_in")


def test_split_pushdown_and_residual(dask_handler) -> None:
    filters = {
        "status__exact": "active",
        "description__icontains": "urgent",
        "created_at__date__gte": "2024-01-01",
    }

    parquet_filters, residual_filters = dask_handler.split_pushdown_and_residual(filters)

    # Check residuals (icontains is not a pushdown op)
    assert residual_filters.get("description__icontains") == "urgent"

    # Check pushdowns
    pushdown_by_field = {f: (op, val) for f, op, val in parquet_filters}
    assert {"status", "created_at"} <= set(pushdown_by_field)

    # Verify the date rewrite to UTC TS
    created_op, created_val = pushdown_by_field["created_at"]
    assert created_op == ">="
    assert isinstance(created_val, pd.Timestamp)
    assert (created_val.tz, created_val) == (dt.UTC, pd.Timestamp("2024-01-01", tz="UTC"))


def test_split_pushdown_and_residual_intercepts_ast_controls(dask_handler) -> None:
    filters = {
        "$or": [
            {"status__exact": "active"},
            {"status__exact": "inactive"},
        ]
    }

    parquet_filters, residual_filters = dask_handler.split_pushdown_and_residual(filters)

    assert parquet_filters == []
    assert residual_filters == filters


def test_ast_compilation(dask_handler) -> None:
    filters = {"$or": [{"price__gt": 100}, {"status__in": ["VIP", "Pro"]}]}

    expr = dask_handler.compile_filters(filters)

    assert isinstance(expr, Or)
    assert isinstance(expr.left, ColOp)
    assert isinstance(expr.right, ColOp)

    assert expr.left.field == "price"
    assert expr.left.op == "gt"
    assert expr.right.field == "status"
    assert expr.right.op == "in"


def test_ast_compilation_and_not(dask_handler) -> None:
    filters = {"$not": {"status__exact": "banned"}}
    expr = dask_handler.compile_filters(filters)
    assert isinstance(expr, Not)
    assert isinstance(expr.inner, ColOp)
    assert expr.inner.op == "exact"
    assert expr.inner.value == "banned"


def test_ast_compilation_implicit_and(dask_handler) -> None:
    filters = {"price__gt": 10, "price__lt": 100}
    expr = dask_handler.compile_filters(filters)

    # Initial state is TrueExpr() & ColOp. And(TrueExpr, ColOp) -> And(And, ColOp)
    # Just asserting it built successfully
    assert isinstance(expr, And)


def test_get_dask_column_rejects_unknown_field(dask_handler) -> None:
    df = dd.from_pandas(pd.DataFrame({"status": ["active"]}), npartitions=1)

    with pytest.raises(AttributeError, match="Field 'missing' not found"):
        dask_handler._get_dask_column(df, "missing", None)


def test_apply_filters_short_circuits_trivial_dask_filters(dask_handler, monkeypatch) -> None:
    dataframe = dd.from_pandas(pd.DataFrame({"status": ["active", "inactive"]}), npartitions=1)
    monkeypatch.setattr(
        TrueExpr,
        "mask",
        lambda self, df: (_ for _ in ()).throw(AssertionError("TrueExpr.mask should not run")),
    )

    result = dask_handler.apply_filters(dataframe, filters={})

    assert result is dataframe
