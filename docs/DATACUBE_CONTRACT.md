# Datacube Contract

`boti-data` supports callable-backed datacube loading through `DatacubeConfig`.
Use `DatacubeContract` when you want typed request hooks and centralized validation.

## Contract Hooks

`DatacubeContract` supports these optional hooks:

- `request_transformer(request) -> request`
- `request_validator(request) -> None`
- `loader(request) -> frame`
- `async_loader(request) -> awaitable[frame]`
- `frame_transformer(frame, request) -> frame`

Supported frame outputs are:

- `pandas.DataFrame`
- `dask.dataframe.DataFrame`
- `pyarrow.Table`
- `polars.DataFrame`

## Execution Order

At runtime, `DatacubeResource` applies hooks in this order:

1. apply `default_cube` when request has no explicit `cube`
2. run `request_transformer` (if set)
3. run `request_validator` (if set)
4. execute loader (`contract.async_loader` / `contract.loader` first, then config loaders)
5. run `frame_transformer` (if set)

Loader precedence:

- sync path: `contract.loader` -> `DatacubeConfig.loader`
- async path: `contract.async_loader` -> `DatacubeConfig.async_loader` -> sync fallback in thread

## Validation Failures

If `request_validator` raises, `boti-data` surfaces an actionable error message with request context:

- selected `cube`
- list of filter keys
- guidance to fix validator inputs or validator logic

Example message shape:

`Datacube contract request validation failed: cube must be 'orders'. Request context cube='inventory', filter_keys=['status__exact']. Adjust DatacubeContract.request_validator or pass valid cube/filter inputs.`

For end-to-end SQL/datacube defensive defaults, see `docs/SECURITY_HARDENING.md`.

## Minimal Pattern

```python
import pandas as pd

from boti_data import DatacubeConfig, DatacubeContract, DataGateway


def loader(request):
    return pd.DataFrame({"id": [1], "status": ["active"]})


def validator(request):
    if request.cube != "orders":
        raise ValueError("cube must be 'orders'")


contract = DatacubeContract(
    loader=loader,
    request_validator=validator,
    request_transformer=lambda request: request.model_copy(
        update={"cube": request.cube or "orders"}
    ),
)

with DataGateway(DatacubeConfig(contract=contract)) as gateway:
    frame = gateway.load(return_type="pandas", status__exact="active")
```

