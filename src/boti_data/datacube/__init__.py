from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from boti_data.datacube.artifact import BaseArtifact
    from boti_data.datacube.base import BaseDataCube
    from boti_data.datacube.contract import (
        DatacubeAsyncLoader,
        DatacubeConfig,
        DatacubeContract,
        DatacubeFrame,
        DatacubeFrameTransformer,
        DatacubeRequestTransformer,
        DatacubeRequestValidator,
        DatacubeSyncLoader,
    )
    from boti_data.datacube.resource import DatacubeResource

__all__ = [
    "BaseDataCube",
    "BaseArtifact",
    "DatacubeAsyncLoader",
    "DatacubeConfig",
    "DatacubeContract",
    "DatacubeFrame",
    "DatacubeFrameTransformer",
    "DatacubeResource",
    "DatacubeRequestTransformer",
    "DatacubeRequestValidator",
    "DatacubeSyncLoader",
]
