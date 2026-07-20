from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data import DatacubeConfig, DatacubeContract, DataHelper
from boti_data.datacube.contract import DatacubeRequest
from boti_data.gateway import DataGateway


def _apply_simple_filters(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    filtered = frame
    for key, value in filters.items():
        if key.endswith("__exact"):
            column = key.removesuffix("__exact")
            filtered = filtered[filtered[column] == value]
        elif key.endswith("__in"):
            column = key.removesuffix("__in")
            filtered = filtered[filtered[column].isin(value)]
    return filtered


def test_datacube_structured_load_supports_dask_return_type() -> None:
    source = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
        }
    )

    def loader(request) -> pd.DataFrame:
        frame = _apply_simple_filters(source, request.filters)
        if request.limit is not None:
            frame = frame.head(request.limit)
        return frame

    with DataGateway(DatacubeConfig(loader=loader)) as gateway:
        frame = gateway.load(return_type="dask", filters={"status__exact": "active"})

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 3]


def test_datacube_configured_mode_merges_sticky_and_runtime_filters() -> None:
    captured_requests = []

    def loader(request) -> pd.DataFrame:
        captured_requests.append(request)
        return pd.DataFrame({"id": [1], "status": ["active"], "region": ["na"]})

    with DataGateway(
        DatacubeConfig(loader=loader),
        table="orders_cube",
        sticky_filters={"status__exact": "active"},
    ) as gateway:
        frame = gateway.load(
            region__exact="na",
            cube="orders_v2",
            return_type="pandas",
        )

    assert isinstance(frame, pd.DataFrame)
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.cube == "orders_v2"
    assert request.filters == {"status__exact": "active", "region__exact": "na"}


@pytest.mark.asyncio
async def test_datacube_async_loader_is_used_when_available() -> None:
    calls: list[str] = []

    async def async_loader(request) -> pd.DataFrame:
        calls.append(request.cube or "")
        return pd.DataFrame({"id": [1, 2], "status": ["active", "active"]})

    async with DataGateway(
        DatacubeConfig(async_loader=async_loader, default_cube="users"),
    ) as gateway:
        frame = await gateway.aload(return_type="pandas")

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 2]
    assert calls == ["users"]


def test_datacube_from_config_supports_backend_mapping() -> None:
    def loader(request) -> pd.DataFrame:
        return pd.DataFrame({"cube": [request.cube or "default"]})

    with DataGateway.from_config(
        {
            "backend": "datacube",
            "loader": loader,
            "default_cube": "inventory",
        }
    ) as gateway:
        frame = gateway.load(return_type="pandas")

    assert isinstance(frame, pd.DataFrame)
    assert frame["cube"].tolist() == ["inventory"]


def test_datacube_helper_views_apply_engine_defaults() -> None:
    def loader(request) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]})

    helper = DataHelper({"backend": "datacube", "loader": loader})
    try:
        frame = helper.dask.load()
    finally:
        helper.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 2]


def test_datacube_contract_hooks_apply_transform_and_validation() -> None:
    seen_cubes: list[str] = []

    def request_transformer(request) -> DatacubeRequest:
        return request.model_copy(update={"cube": request.cube or "orders_v2"})

    def request_validator(request) -> None:
        if request.cube != "orders_v2":
            raise ValueError("unexpected cube")

    def loader(request) -> pd.DataFrame:
        seen_cubes.append(request.cube)
        return pd.DataFrame({"id": [1], "status": ["active"]})

    def frame_transformer(frame: pd.DataFrame, request) -> pd.DataFrame:
        frame = frame.copy()
        frame["cube"] = request.cube
        return frame

    contract = DatacubeContract(
        loader=loader,
        request_transformer=request_transformer,
        request_validator=request_validator,
        frame_transformer=frame_transformer,
    )

    with DataGateway(DatacubeConfig(contract=contract)) as gateway:
        frame = gateway.load(return_type="pandas")

    assert isinstance(frame, pd.DataFrame)
    assert frame["cube"].tolist() == ["orders_v2"]
    assert seen_cubes == ["orders_v2"]


@pytest.mark.asyncio
async def test_datacube_contract_async_loader_takes_precedence() -> None:
    async def async_loader(request) -> pd.DataFrame:
        return pd.DataFrame({"cube": [request.cube or "none"]})

    contract = DatacubeContract(async_loader=async_loader)

    async with DataGateway(DatacubeConfig(contract=contract, default_cube="async_cube")) as gateway:
        frame = await gateway.aload(return_type="pandas")

    assert isinstance(frame, pd.DataFrame)
    assert frame["cube"].tolist() == ["async_cube"]


def test_datacube_config_requires_loader_source() -> None:
    with pytest.raises(ValueError, match="DatacubeConfig requires loader/async_loader"):
        DatacubeConfig()


def test_datacube_contract_validator_error_includes_actionable_context() -> None:
    def validator(request) -> None:
        if request.cube != "orders":
            raise ValueError("cube must be 'orders'")

    contract = DatacubeContract(
        loader=lambda _request: pd.DataFrame({"id": [1]}),
        request_validator=validator,
    )

    with DataGateway(DatacubeConfig(contract=contract)) as gateway:
        with pytest.raises(
            ValueError, match="Datacube contract request validation failed"
        ) as exc_info:
            gateway.load(return_type="pandas", cube="inventory", status__exact="active")

    message = str(exc_info.value)
    assert "cube must be 'orders'" in message
    assert "cube='inventory'" in message
    assert "filter_keys=['status__exact']" in message
