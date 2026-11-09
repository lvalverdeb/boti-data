# Datacube backend

`DataGateway` also supports a callable-backed `datacube` backend for in-process cube loaders while keeping the same return-type API (`pandas`/`dask`/`arrow`/`polars`).

```python
import pandas as pd

from boti_data import DatacubeConfig, DatacubeContract, DataGateway


def loader(request):
    frame = pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]})
    if request.filters.get("status__exact") == "active":
        frame = frame[frame["status"] == "active"]
    return frame


contract = DatacubeContract(
    request_transformer=lambda request: request.model_copy(update={"cube": request.cube or "orders_v2"}),
    frame_transformer=lambda frame, request: frame.assign(cube=request.cube),
)

with DataGateway(
    DatacubeConfig(loader=loader, default_cube="orders", contract=contract),
    table="orders",
    sticky_filters={"status__exact": "active"},
) as gateway:
    df = gateway.load(return_type="pandas")
```

You can also configure this path with `DataGateway.from_config({"backend": "datacube", ...})` or `DataHelper({"backend": "datacube", ...})`.

See [the datacube contract notes](../proposals/DATACUBE_CONTRACT.md) for hook ordering, loader precedence, and validator rejection guidance.
See [the security hardening notes](../proposals/SECURITY_HARDENING.md) for SQL/raw-sql policy and distributed credential hardening defaults.
For a runnable rejection flow, see `examples/data_facade_datacube_contract_rejection.py`.
