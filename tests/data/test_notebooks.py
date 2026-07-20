from __future__ import annotations

import json
from pathlib import Path


def _notebook_source_text(notebook_name: str) -> str:
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / notebook_name
    assert notebook_path.exists()

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("nbformat") == 4
    cells = payload.get("cells", [])
    assert isinstance(cells, list)
    assert cells, f"{notebook_name} should include at least one cell."

    return "\n".join("".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict))


def test_hybrid_dataset_notebook_exists_and_mentions_hybrid_dataset() -> None:
    source_text = _notebook_source_text("09_hybrid_dataset.ipynb")
    assert "HybridDataset" in source_text
    assert "historical" in source_text
    assert "live" in source_text


def test_enrichment_notebook_mentions_distributed_enricher_apis() -> None:
    source_text = _notebook_source_text("11_enrichment_v1.ipynb")
    assert "LocalCluster" in source_text
    assert "DataHelper.session" in source_text
    assert "AsyncFrameEnricher" in source_text
    assert "AttachmentSpec" in source_text


def test_pipelines_notebook_mentions_distributed_pipeline_apis() -> None:
    source_text = _notebook_source_text("12_pipelines.ipynb")
    assert "LocalCluster" in source_text
    assert "DataHelper.session" in source_text
    assert "SinkPipeline" in source_text
    assert "ParquetPipeline" in source_text


def test_incremental_loading_notebook_mentions_incremental_apis() -> None:
    source_text = _notebook_source_text("21_incremental_loading.ipynb")
    assert "DataHelper.load_incremental" in source_text or "load_incremental" in source_text
    assert "FileWatermarkStore" in source_text
    assert "IncrementalResult" in source_text
    assert "watermark_field" in source_text
    assert "watermark_store" in source_text


def test_incremental_pipeline_notebook_mentions_incremental_apis() -> None:
    source_text = _notebook_source_text("22_incremental_pipeline.ipynb")
    assert "ParquetPipeline.incremental" in source_text or "incremental" in source_text
    assert "ParquetPipeline" in source_text
    assert "watermark_field" in source_text
    assert "materialize" in source_text
    assert "FileWatermarkStore" in source_text
