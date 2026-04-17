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


def test_diagnostics_example_runs_and_returns_summary(capsys, monkeypatch):
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


def test_distributed_parquet_example_runs_and_returns_summary(capsys):
    module = _load_example_module("data_facade_distributed_parquet.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["matched_rows"] == 3
    assert result["unmatched_rows"] == 1
    assert "Scheduler session client:" in output
    assert "Joined parquet head:" in output


def test_helper_legacy_example_runs_and_returns_summary(capsys):
    module = _load_example_module("data_helper_legacy.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["rows"] == 2
    assert result["records"][0]["barcode"] == "A-001"
    assert result["records"][1]["barcode"] == "A-002"
    assert "Legacy-config helper records:" in output


def test_helper_distributed_example_runs_and_returns_summary(capsys):
    module = _load_example_module("data_helper_distributed.py")

    result = module.main()
    output = capsys.readouterr().out

    assert result["workers"] == 2
    assert result["shared_client_reused"] is True
    assert result["dry_run_partitions"] >= 1
    assert len(result["preview_records"]) == 2
    assert result["matched_rows"] == 3
    assert result["unmatched_rows"] == 1
    assert "Distributed helper workers:" in output
    assert "Shared session reused: True" in output
    assert "Dry-run partitions:" in output
    assert "Helper preview records:" in output
    assert "Joined helper records:" in output


