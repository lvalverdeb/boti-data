"""Tests for BaseArtifact — date window + before/after load hooks."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

import pandas as pd
import pytest

from boti_data.datacube.artifact import BaseArtifact, _validate_and_format_date
from boti_data.datacube.contract import DatacubeConfig
from boti_data.gateway import DataGateway
from boti_data.helper import DataHelper

_RT = "pandas"


def _make_helper(loader: Any) -> DataHelper:
    return DataHelper.from_gateway(DataGateway(DatacubeConfig(loader=loader)))


def _pandas_loader(request: Any) -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]})


def _empty_loader(request: Any) -> pd.DataFrame:
    return pd.DataFrame({"id": [], "status": []})


def _run_async(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ── _validate_and_format_date ────────────────────────────────────────────────


class TestValidateAndFormatDate:
    def test_none_returns_none(self) -> None:
        assert _validate_and_format_date("test", None) is None

    def test_string_date_normalized(self) -> None:
        assert _validate_and_format_date("test", "2024-03-15") == "2024-03-15"

    def test_string_date_with_time_normalized(self) -> None:
        assert _validate_and_format_date("test", "2024-03-15T10:30:00") == "2024-03-15"

    def test_date_object_normalized(self) -> None:
        assert _validate_and_format_date("test", date(2024, 3, 15)) == "2024-03-15"

    def test_datetime_object_normalized(self) -> None:
        assert _validate_and_format_date("test", datetime(2024, 3, 15, 10, 30)) == "2024-03-15"

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be a valid date"):
            _validate_and_format_date("test", "not-a-date")

    def test_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be str, date, datetime"):
            _validate_and_format_date("test", 123)


# ── before_load / after_load hooks ──────────────────────────────────────────


class TestLoadHooks:
    def test_before_load_and_after_load_called_on_load(self) -> None:
        call_order: list[str] = []

        class MyArtifact(BaseArtifact):
            def before_load(self, **kwargs: Any) -> None:
                call_order.append("before")

            def after_load(self, **kwargs: Any) -> None:
                call_order.append("after")

        artifact = MyArtifact.from_helper(_make_helper(_pandas_loader))
        artifact.load(return_type=_RT)

        assert call_order == ["before", "after"]

    def test_hooks_not_called_when_not_overridden(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        result = artifact.load(return_type=_RT)
        assert isinstance(result, pd.DataFrame)

    def test_abefore_load_and_aafter_load_called_on_aload(self) -> None:
        call_order: list[str] = []

        class MyArtifact(BaseArtifact):
            async def abefore_load(self, **kwargs: Any) -> None:
                call_order.append("abefore")

            async def aafter_load(self, **kwargs: Any) -> None:
                call_order.append("aafter")

        artifact = MyArtifact.from_helper(_make_helper(_pandas_loader))
        _run_async(artifact.aload(return_type=_RT))

        assert call_order == ["abefore", "aafter"]

    def test_aload_hooks_not_called_when_not_overridden(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        result = _run_async(artifact.aload(return_type=_RT))
        assert isinstance(result, pd.DataFrame)

    def test_after_load_can_modify_df(self) -> None:
        class FilterArtifact(BaseArtifact):
            def after_load(self, **kwargs: Any) -> None:
                if self.df is not None:
                    self.df = self.df[self.df["status"] == "active"]

        artifact = FilterArtifact.from_helper(_make_helper(_pandas_loader))
        result = artifact.load(return_type=_RT)

        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 3]


# ── date window ──────────────────────────────────────────────────────────────


class TestDateWindow:
    def test_no_dates_by_default(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        assert artifact.parquet_start_date is None
        assert artifact.parquet_end_date is None
        assert artifact.has_date_window() is False
        assert artifact.date_window() == (None, None)

    def test_dates_normalized_to_strings(self) -> None:
        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            parquet_start_date="2024-01-15",
            parquet_end_date="2024-06-30",
        )
        assert artifact.parquet_start_date == "2024-01-15"
        assert artifact.parquet_end_date == "2024-06-30"
        assert artifact.has_date_window() is True
        assert artifact.date_window() == ("2024-01-15", "2024-06-30")

    def test_date_objects_normalized(self) -> None:
        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            parquet_start_date=date(2024, 1, 15),
            parquet_end_date=date(2024, 6, 30),
        )
        assert artifact.parquet_start_date == "2024-01-15"
        assert artifact.parquet_end_date == "2024-06-30"

    def test_start_after_end_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be after"):
            BaseArtifact.from_helper(
                _make_helper(_pandas_loader),
                parquet_start_date="2024-06-30",
                parquet_end_date="2024-01-15",
            )

    def test_only_start_date_is_valid(self) -> None:
        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            parquet_start_date="2024-01-15",
        )
        assert artifact.has_date_window() is True

    def test_only_end_date_is_valid(self) -> None:
        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            parquet_end_date="2024-06-30",
        )
        assert artifact.has_date_window() is True


# ── .df stores loaded frame ─────────────────────────────────────────────────


class TestDfStorage:
    def test_df_stores_frame_after_load(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        result = artifact.load(return_type=_RT)
        assert artifact.df is result

    def test_df_stores_frame_after_aload(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        result = _run_async(artifact.aload(return_type=_RT))
        assert artifact.df is result

    def test_df_stores_empty_frame_when_empty_load(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_empty_loader))
        artifact.load(return_type=_RT)
        assert artifact._has_data() is False


# ── to_params ────────────────────────────────────────────────────────────────


class TestToParams:
    def test_to_params_returns_dict(self) -> None:
        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            parquet_start_date="2024-01-01",
            parquet_end_date="2024-12-31",
        )
        params = artifact.to_params()

        assert params["parquet_start_date"] == "2024-01-01"
        assert params["parquet_end_date"] == "2024-12-31"
        assert params["data_wrapper_class"] is None
        assert "class_params" in params

    def test_to_params_with_data_wrapper_class(self) -> None:
        class MyWrapper:
            pass

        artifact = BaseArtifact.from_helper(
            _make_helper(_pandas_loader),
            data_wrapper_class=MyWrapper,
        )
        params = artifact.to_params()
        assert params["data_wrapper_class"] is MyWrapper


# ── config ───────────────────────────────────────────────────────────────────


class TestConfig:
    def test_config_class_var_used_as_defaults(self) -> None:
        class MyArtifact(BaseArtifact):
            config = {"parquet_storage_path": "/data/test"}

        assert MyArtifact.config["parquet_storage_path"] == "/data/test"

    def test_config_not_mutated_across_instances(self) -> None:
        class MyArtifact(BaseArtifact):
            config = {"key": "original"}

        original = MyArtifact.config.copy()
        MyArtifact.from_helper(_make_helper(_pandas_loader))
        assert MyArtifact.config == original


# ── from_helper ──────────────────────────────────────────────────────────────


class TestFromHelper:
    def test_from_helper_creates_instance(self) -> None:
        helper = _make_helper(_pandas_loader)
        artifact = BaseArtifact.from_helper(helper)
        assert artifact._helper is helper
        assert artifact.df is None

    def test_from_helper_overrides_config(self) -> None:
        helper = _make_helper(_pandas_loader)
        artifact = BaseArtifact.from_helper(helper, custom="value")
        assert artifact.config["custom"] == "value"


# ── logger kwarg is honored, not always overwritten ──────────────────────────


class TestLoggerKwarg:
    def test_init_honors_passed_in_logger(self, monkeypatch: Any) -> None:
        sentinel = object()
        monkeypatch.setattr("boti_data.helper.DataHelper.__init__", lambda self, *a, **kw: None)
        artifact = BaseArtifact(logger=sentinel)
        assert artifact.logger is sentinel
        # class_params defaults to {"logger": self.logger} when not passed explicitly.
        assert artifact.class_params["logger"] is sentinel

    def test_init_defaults_logger_when_not_passed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("boti_data.helper.DataHelper.__init__", lambda self, *a, **kw: None)
        artifact = BaseArtifact()
        assert artifact.logger is not None
        assert artifact.logger.__class__.__name__ == "Logger"

    def test_from_helper_honors_passed_in_logger(self) -> None:
        sentinel = object()
        helper = _make_helper(_pandas_loader)
        artifact = BaseArtifact.from_helper(helper, logger=sentinel)
        assert artifact.logger is sentinel
        assert artifact.class_params["logger"] is sentinel

    def test_from_helper_defaults_logger_when_not_passed(self) -> None:
        helper = _make_helper(_pandas_loader)
        artifact = BaseArtifact.from_helper(helper)
        assert artifact.logger is not None
        assert artifact.logger.__class__.__name__ == "Logger"


# ── context manager ──────────────────────────────────────────────────────────


class TestLifecycle:
    def test_context_manager_enter_exit(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        with artifact:
            result = artifact.load(return_type=_RT)
            assert len(result) == 3

    def test_context_manager_returns_self(self) -> None:
        artifact = BaseArtifact.from_helper(_make_helper(_pandas_loader))
        with artifact as ctx:
            assert ctx is artifact
