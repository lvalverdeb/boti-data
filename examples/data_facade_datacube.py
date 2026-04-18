"""
DataGateway example for callable-backed datacube loading.
"""

from __future__ import annotations

import pandas as pd

from boti_data.datacube import DatacubeConfig, DatacubeContract
from boti_data.gateway import DataGateway


def datacube_loader(request):
    source = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
            "region": ["na", "na", "eu"],
        }
    )
    frame = source
    for key, value in request.filters.items():
        if key.endswith("__exact"):
            frame = frame[frame[key.removesuffix("__exact")] == value]
    if request.columns:
        frame = frame[request.columns]
    if request.limit is not None:
        frame = frame.head(request.limit)
    return frame


def main() -> None:
    contract = DatacubeContract(
        request_transformer=lambda request: request.model_copy(
            update={"cube": request.cube or "orders_v2"}
        ),
        frame_transformer=lambda frame, request: frame.assign(cube=request.cube),
    )
    config = DatacubeConfig(loader=datacube_loader, default_cube="orders", contract=contract)

    with DataGateway(config, table="orders", sticky_filters={"status__exact": "active"}) as gateway:
        dask_frame = gateway.load(return_type="dask", region__exact="na")
        pandas_frame = gateway.load(return_type="pandas", region__exact="na", columns=["id", "status"])

    print("dask")
    print(dask_frame.compute())
    print("\npandas")
    print(pandas_frame)


if __name__ == "__main__":
    main()

