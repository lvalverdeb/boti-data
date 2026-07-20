"""Tests for FileWatermarkStore and build_incremental_filters.

Split out of test_watermark.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import json
import threading

from boti_data import FileWatermarkStore, build_incremental_filters

# ---------------------------------------------------------------------------
# WatermarkStore
# ---------------------------------------------------------------------------


def test_file_store_read_write_roundtrip(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value="2026-05-10")
    assert store.read(source="test") == "2026-05-10"


def test_file_store_clear(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="t1", value=100)
    store.write(source="t2", value=200)
    store.clear(source="t1")
    assert store.read(source="t1") is None
    assert store.read(source="t2") == 200


def test_file_store_missing_key(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    assert store.read(source="nonexistent") is None


def test_file_store_nonexistent_file(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "no_such_file.json"))
    assert store.read(source="x") is None


def test_file_store_overwrite_value(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="k", value="first")
    store.write(source="k", value="second")
    assert store.read(source="k") == "second"


def test_file_store_multiple_sources(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="a", value=1)
    store.write(source="b", value=2)
    store.write(source="c", value=3)
    assert store.read(source="a") == 1
    assert store.read(source="b") == 2
    assert store.read(source="c") == 3


def test_file_store_clear_nonexistent(tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.clear(source="nobody")  # should not raise


def test_file_store_concurrent_access(tmp_path) -> None:
    """Basic thread-safety smoke test."""
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(50):
                store.write(source="concurrent", value=i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.read(source="concurrent") is not None


def test_file_store_persists_to_disk(tmp_path) -> None:
    path = tmp_path / "persist.json"
    store = FileWatermarkStore(str(path))
    store.write(source="disk", value="hello")
    raw = json.loads(path.read_text())
    assert raw["disk"] == "hello"


# ---------------------------------------------------------------------------
# build_incremental_filters
# ---------------------------------------------------------------------------


def test_build_incremental_filters_gt() -> None:
    result = build_incremental_filters(
        watermark_field="updated_at",
        watermark_value="2026-05-01",
        operator="gt",
    )
    assert result == {"updated_at__gt": "2026-05-01"}


def test_build_incremental_filters_gte() -> None:
    result = build_incremental_filters(
        watermark_field="event_date",
        watermark_value="2026-05-01",
        operator="gte",
    )
    assert result == {"event_date__gte": "2026-05-01"}


def test_build_incremental_filters_default_operator() -> None:
    result = build_incremental_filters(
        watermark_field="id",
        watermark_value=100,
    )
    assert result == {"id__gt": 100}
