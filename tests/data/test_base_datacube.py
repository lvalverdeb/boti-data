"""Tests for BaseDataCube — inheritance-based datacube with transform hooks."""

from __future__ import annotations

import asyncio
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from boti_data.datacube.base import BaseDataCube
from boti_data.datacube.contract import DatacubeConfig
from boti_data.gateway import DataGateway
from boti_data.helper import DataHelper

_RT = "pandas"


def _make_helper(loader: Any) -> DataHelper:
    """Build a DataHelper backed by a simple callable loader."""
    return DataHelper.from_gateway(DataGateway(DatacubeConfig(loader=loader)))


def _pandas_loader(request: Any) -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]})


def _empty_loader(request: Any) -> pd.DataFrame:
    return pd.DataFrame({"id": [], "status": []})


def _run_async(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ── fix_data / afix_data hook dispatch ──────────────────────────────────────


class TestFixDataHook:
    def test_fix_data_called_on_load_when_overridden(self) -> None:
        class MyCube(BaseDataCube):
            def fix_data(self, **kwargs: Any) -> None:
                self.df = self.df[self.df["status"] == "active"]

        cube = MyCube.from_helper(_make_helper(_pandas_loader))
        result = cube.load(return_type=_RT)

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 3]

    def test_fix_data_not_called_when_not_overridden(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        result = cube.load(return_type=_RT)

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 2, 3]

    def test_afix_data_called_on_aload_when_overridden(self) -> None:
        class MyCube(BaseDataCube):
            async def afix_data(self, **kwargs: Any) -> None:
                self.df = self.df[self.df["id"] > 1]

        cube = MyCube.from_helper(_make_helper(_pandas_loader))
        result = _run_async(cube.aload(return_type=_RT))

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [2, 3]

    def test_aload_falls_back_to_fix_data_when_afix_not_overridden(self) -> None:
        class SyncOnlyCube(BaseDataCube):
            def fix_data(self, **kwargs: Any) -> None:
                self.df = self.df[self.df["id"] == 2]

        cube = SyncOnlyCube.from_helper(_make_helper(_pandas_loader))
        result = _run_async(cube.aload(return_type=_RT))

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [2]

    def test_no_hooks_called_when_neither_overridden(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        result = _run_async(cube.aload(return_type=_RT))

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 2, 3]


# ── _has_data ───────────────────────────────────────────────────────────────


class TestHasData:
    def test_has_data_false_when_df_is_none(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        assert cube._has_data() is False

    def test_has_data_true_for_pandas(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = pd.DataFrame({"a": [1, 2]})
        assert cube._has_data() is True

    def test_has_data_false_for_empty_pandas(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = pd.DataFrame({"a": []})
        assert cube._has_data() is False

    def test_has_data_true_for_dask(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = dd.from_pandas(pd.DataFrame({"a": [1, 2]}), npartitions=1)
        assert cube._has_data() is True

    def test_has_data_false_for_empty_dask(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = dd.from_pandas(pd.DataFrame({"a": []}), npartitions=1)
        assert cube._has_data() is False

    def test_has_data_true_for_pyarrow(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = pa.table({"a": [1, 2]})
        assert cube._has_data() is True

    def test_has_data_false_for_empty_pyarrow(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = pa.table({"a": pa.array([], type=pa.int64())})
        assert cube._has_data() is False

    def test_has_data_true_for_polars(self) -> None:
        import polars as pl

        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        cube.df = pl.DataFrame({"a": [1, 2]})
        assert cube._has_data() is True


# ── .df stores loaded frame ─────────────────────────────────────────────────


class TestDfStorage:
    def test_df_stores_frame_after_load(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        result = cube.load(return_type=_RT)
        assert cube.df is result

    def test_df_stores_frame_after_aload(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        result = _run_async(cube.aload(return_type=_RT))
        assert cube.df is result

    def test_df_stores_empty_frame_when_empty_load(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_empty_loader))
        result = cube.load(return_type=_RT)
        assert cube._has_data() is False


# ── config class variable merged with kwargs ─────────────────────────────────


class TestConfig:
    def test_config_class_var_used_as_defaults(self) -> None:
        class MyCube(BaseDataCube):
            config = {"backend": "sqlalchemy", "query_only": True}

        assert MyCube.config["backend"] == "sqlalchemy"

    def test_config_not_mutated_across_instances(self) -> None:
        class MyCube(BaseDataCube):
            config = {"key": "original"}

        original = MyCube.config.copy()
        instance = MyCube.from_helper(_make_helper(_pandas_loader))
        instance.config = {**MyCube.config, "extra": "value"}
        assert MyCube.config == original


# ── from_helper classmethod ──────────────────────────────────────────────────


class TestFromHelper:
    def test_from_helper_creates_instance(self) -> None:
        helper = _make_helper(_pandas_loader)
        cube = BaseDataCube.from_helper(helper)
        assert cube._helper is helper
        assert cube.df is None

    def test_from_helper_overrides_config(self) -> None:
        helper = _make_helper(_pandas_loader)
        cube = BaseDataCube.from_helper(helper, custom="value")
        assert cube.config["custom"] == "value"


# ── logger kwarg is honored, not always overwritten ──────────────────────────


class TestLoggerKwarg:
    def test_init_honors_passed_in_logger(self, monkeypatch: Any) -> None:
        sentinel = object()
        monkeypatch.setattr("boti_data.helper.DataHelper.__init__", lambda self, *a, **kw: None)
        cube = BaseDataCube(logger=sentinel)
        assert cube.logger is sentinel

    def test_init_defaults_logger_when_not_passed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("boti_data.helper.DataHelper.__init__", lambda self, *a, **kw: None)
        cube = BaseDataCube()
        assert cube.logger is not None
        assert cube.logger.__class__.__name__ == "Logger"

    def test_from_helper_honors_passed_in_logger(self) -> None:
        sentinel = object()
        helper = _make_helper(_pandas_loader)
        cube = BaseDataCube.from_helper(helper, logger=sentinel)
        assert cube.logger is sentinel

    def test_from_helper_defaults_logger_when_not_passed(self) -> None:
        helper = _make_helper(_pandas_loader)
        cube = BaseDataCube.from_helper(helper)
        assert cube.logger is not None
        assert cube.logger.__class__.__name__ == "Logger"


# ── context manager ──────────────────────────────────────────────────────────


class TestLifecycle:
    def test_context_manager_enter_exit(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        with cube:
            result = cube.load(return_type=_RT)
            assert len(result) == 3

    def test_context_manager_returns_self(self) -> None:
        cube = BaseDataCube.from_helper(_make_helper(_pandas_loader))
        with cube as ctx:
            assert ctx is cube
