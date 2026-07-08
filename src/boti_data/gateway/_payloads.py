from __future__ import annotations

from typing import Any

from boti_data.gateway.load_request import GatewayLoadRequest
from boti_data.gateway.normalization import split_control_and_filters
from boti_data.gateway.requests import ResolvedReturnType


def _extract_options_dict(
    source: GatewayLoadRequest | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(source, GatewayLoadRequest):
        return source.build_loader_options()
    return source


def structured_sql_request_payload(
    options: GatewayLoadRequest | dict[str, Any],
    *,
    return_type: ResolvedReturnType,
) -> dict[str, Any]:
    src = _extract_options_dict(options)
    return {
        "sql": src.get("sql"),
        "statement": src.get("statement"),
        "model": src.get("model"),
        "filters": src.get("filters", {}),
        "params": src.get("params", {}),
        "limit": src.get("limit"),
        "columns": src.get("columns"),
        "as_pandas": return_type == "pandas" or bool(src.get("as_pandas", False)),
        "diagnostics": bool(src.get("diagnostics", False)),
        "return_type": return_type,
    }


def structured_parquet_request_payload(
    options: GatewayLoadRequest | dict[str, Any],
    *,
    return_type: ResolvedReturnType,
) -> dict[str, Any]:
    src = _extract_options_dict(options)
    _ctrl, runtime_filters = split_control_and_filters(
        options if isinstance(options, dict) else options.model_dump()
    )
    merged_filters = {**runtime_filters, **src.get("filters", {})}
    return {
        "filters": merged_filters,
        "raw_filters": src.get("raw_filters"),
        "limit": src.get("limit"),
        "columns": src.get("columns"),
        "as_pandas": return_type == "pandas" or bool(src.get("as_pandas", False)),
        "diagnostics": bool(src.get("diagnostics", False)),
        "return_type": return_type,
    }


def datacube_request_from_options(opts: GatewayLoadRequest | dict[str, Any]) -> Any:
    from boti_data.datacube.contract import DatacubeRequest

    if isinstance(opts, GatewayLoadRequest):
        _ctrl, runtime_filters = split_control_and_filters(opts.model_dump())
        return DatacubeRequest(
            filters=runtime_filters | opts.filters,
            limit=opts.limit,
            cube=opts.cube,
            return_type=opts.return_type,
        )
    _ctrl, runtime_filters = split_control_and_filters(opts)
    merged_filters = {**runtime_filters, **opts.get("filters", {})}
    return DatacubeRequest(
        filters=merged_filters,
        limit=opts.get("limit"),
        cube=opts.get("cube"),
        return_type=opts.get("return_type"),
    )
