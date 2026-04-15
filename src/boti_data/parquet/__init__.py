"""
Parquet-backed data resources.
"""

from boti_data.parquet.reader import ParquetReader
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

__all__ = ["ParquetDataConfig", "ParquetDataResource", "ParquetReader"]
