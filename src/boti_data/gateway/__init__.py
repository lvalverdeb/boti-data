from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from boti_data.db import AsyncSqlDatabaseResource

    from .core import DataGateway
    from .policies import GatewayPolicies
    from .requests import (
        DataFrameOptions,
        DataFrameParams,
        ParquetLoadRequest,
        SqlLoadRequest,
        TableDescription,
    )

__all__ = [
    "AsyncSqlDatabaseResource",
    "DataFrameOptions",
    "DataFrameParams",
    "DataGateway",
    "GatewayPolicies",
    "ParquetLoadRequest",
    "SqlLoadRequest",
    "TableDescription",
]
