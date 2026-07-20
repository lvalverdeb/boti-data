from __future__ import annotations

import pytest

from boti_data import SinkRegistry, available_sinks, create_sink, register_sink
from boti_data.pipelines.sinks import CsvSink, JsonlSink


def test_global_sink_registry_exposes_builtin_sinks() -> None:
    names = set(available_sinks())
    assert {"parquet", "csv", "jsonl"}.issubset(names)


def test_global_sink_registry_creates_builtin_sink_instances(temp_project_root) -> None:
    csv_sink = create_sink(
        "csv",
        {
            "storage_path": str(temp_project_root / "registry_csv"),
            "project_root": temp_project_root,
        },
    )
    jsonl_sink = create_sink(
        "jsonl",
        {
            "storage_path": str(temp_project_root / "registry_jsonl"),
            "project_root": temp_project_root,
        },
    )

    try:
        assert isinstance(csv_sink, CsvSink)
        assert isinstance(jsonl_sink, JsonlSink)
    finally:
        csv_sink.close()
        jsonl_sink.close()


def test_sink_registry_register_and_overwrite_behaviour() -> None:
    registry = SinkRegistry()

    class DummySink:
        def __init__(self, config):
            self.config = config

        def write(self, *args, **kwargs) -> None:
            raise NotImplementedError

        async def awrite(self, *args, **kwargs) -> None:
            raise NotImplementedError

    registry.register("dummy", lambda config: DummySink(config))
    sink = registry.create("dummy", {"value": 1})
    assert isinstance(sink, DummySink)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", lambda config: DummySink(config))

    registry.register("dummy", lambda config: DummySink({"overwritten": True}), overwrite=True)
    overwritten = registry.create("dummy", {"value": 2})
    assert getattr(overwritten, "config") == {"overwritten": True}


def test_register_sink_global_helper_supports_custom_factory() -> None:
    class TinySink:
        def __init__(self, config):
            self.config = config

        def write(self, *args, **kwargs) -> None:
            raise NotImplementedError

        async def awrite(self, *args, **kwargs) -> None:
            raise NotImplementedError

    name = "tiny_test_sink"
    register_sink(name, lambda config: TinySink(config), overwrite=True)
    sink = create_sink(name, {"ok": True})
    assert isinstance(sink, TinySink)
