from __future__ import annotations

import json
from pathlib import Path


def test_hybrid_dataset_notebook_exists_and_mentions_hybrid_dataset():
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "09_hybrid_dataset.ipynb"
    assert notebook_path.exists()

    raw_text = notebook_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    cells = payload.get("cells", []) if isinstance(payload, dict) else []
    all_source = "\n".join("".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict))

    source_text = all_source if all_source else raw_text
    assert "HybridDataset" in source_text
    assert "historical" in source_text
    assert "live" in source_text

