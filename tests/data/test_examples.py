from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_example_module(filename: str):
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostics_example_runs_and_returns_summary(capsys, monkeypatch) -> None:
    monkeypatch.setenv("BOTI_EXAMPLE_DIAGNOSTIC_LEFT_ROWS", "1000")
    monkeypatch.setenv("BOTI_EXAMPLE_DIAGNOSTIC_RIGHT_ROWS", "750")
    monkeypatch.setenv("BOTI_EXAMPLE_DIAGNOSTIC_BATCH_SIZE", "250")
    monkeypatch.setenv("BOTI_EXAMPLE_DIAGNOSTIC_WORKERS", "1")
    module = _load_example_module("data_facade_diagnostics.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["user_rows"] == 1000
    assert result["profile_rows"] == 750
    assert result["matched_rows"] == 750
    assert result["unmatched_rows"] == 250
    assert result["status_counts"] == {"active": 800, "inactive": 200}
    assert "Managed session client:" in output
    assert "Left rows: 1,000" in output
    assert "Status counts:" in output
    assert "Match summary:" in output


def test_distributed_parquet_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_facade_distributed_parquet.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["matched_rows"] == 3
    assert result["unmatched_rows"] == 1
    assert "Scheduler session client:" in output
    assert "Joined parquet head:" in output


def test_helper_legacy_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_helper_legacy.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["rows"] == 2
    assert result["records"][0]["barcode"] == "A-001"
    assert result["records"][1]["barcode"] == "A-002"
    assert "Legacy-config helper records:" in output


def test_helper_distributed_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_helper_distributed.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["workers"] == 2
    assert result["shared_client_reused"] is True
    assert result["dry_run_partitions"] >= 1
    assert len(result["preview_records"]) == 2
    assert (result["matched_rows"], result["unmatched_rows"]) == (3, 1)
    expected_lines = [
        "Distributed helper workers:",
        "Shared session reused: True",
        "Dry-run partitions:",
        "Helper preview records:",
        "Joined helper records:",
    ]
    missing = [line for line in expected_lines if line not in output]
    assert not missing, f"missing lines in output: {missing}"


def test_helper_bootstrap_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_helper_bootstrap.py")

    result = module.main()
    output = capsys.readouterr().out

    engine_views = result["engine_views"]
    assert (
        engine_views["pandas_rows"],
        engine_views["polars_rows"],
        engine_views["dask_rows"],
        result["sync_row_count"],
        result["async_row_count"],
    ) == (4, 4, 4, 4, 4)
    assert result["dry_run_partitions"] >= 1
    assert (result["sync_gather"], result["async_gather"]) == ([4], [4])
    assert (result["probably_empty"], result["is_empty"]) == (False, False)
    assert (
        set(result["unique_values"]["product_type_id"]),
        set(result["unique_values"]["global_track_id"]),
    ) == ({1}, {1, 2, 3, 4})
    expected_lines = [
        "Bootstrap engine-view rows:",
        "Dry-run partitions:",
        "Sync row count: 4",
        "Async row count: 4",
    ]
    missing = [line for line in expected_lines if line not in output]
    assert not missing, f"missing lines in output: {missing}"


def test_hybrid_dataset_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_hybrid_dataset.py")

    result = module.main()
    output = capsys.readouterr().out

    assert (result["sync_type"], result["sync_rows"]) == ("DataFrame", 5)
    assert (result["historical_type"], result["historical_rows"]) == ("DataFrame", 2)
    assert (result["live_type"], result["live_rows"]) == ("DataFrame", 3)
    assert (result["async_type"], result["async_rows"]) == ("DataFrame", 2)
    expected_lines = [
        "Hybrid sync type=DataFrame rows=5",
        "Hybrid historical view type=DataFrame rows=2",
        "Hybrid live view type=DataFrame rows=3",
        "Hybrid async type=DataFrame rows=2",
    ]
    missing = [line for line in expected_lines if line not in output]
    assert not missing, f"missing lines in output: {missing}"


def test_hybrid_dataset_sql_parquet_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_hybrid_dataset_sql_parquet.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["mixed_type"] == "DataFrame"
    assert result["mixed_rows"] == 4
    assert result["eager_type"] == "DataFrame"
    assert result["eager_rows"] == 4
    assert result["total_amount"] == 700
    assert "Hybrid parquet+sql mixed type=DataFrame rows=4" in output
    assert "Hybrid parquet+sql eager type=DataFrame rows=4" in output
    assert "Hybrid parquet+sql total amount=700" in output


def test_hybrid_dataset_distributed_example_runs_and_returns_summary(capsys) -> None:
    module = _load_example_module("data_hybrid_dataset_distributed.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["workers"] == 1
    assert result["sync_type"] == "DataFrame"
    assert result["sync_rows"] == 4
    assert result["async_rows"] == 2
    assert result["ids"] == [1, 2, 10, 11]
    assert result["statuses"] == ["hist", "hist", "live", "live"]
    assert "Hybrid distributed workers: 1" in output
    assert "Hybrid distributed sync type=DataFrame rows=4" in output
    assert "Hybrid distributed async rows=2" in output


def test_datacube_contract_rejection_example_reports_actionable_error(capsys) -> None:
    module = _load_example_module("data_facade_datacube_contract_rejection.py")

    result = module.main()
    output = capsys.readouterr().out

    assert "Expected contract validation error:" in output
    assert "Datacube contract request validation failed" in result["error"]
    assert "cube='inventory'" in result["error"]
    assert "filter_keys=['status__exact']" in result["error"]
