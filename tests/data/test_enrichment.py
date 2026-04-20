from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data import AsyncFrameEnricher, AttachmentSpec


@pytest.fixture
def base_frame() -> dd.DataFrame:
    return dd.from_pandas(
        pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "status": ["active", "inactive", "active"],
            }
        ),
        npartitions=1,
    )


@pytest.mark.asyncio
async def test_async_frame_enricher_enriches_selected_spec(base_frame):
    async def attachment_fn(ids):
        return pd.DataFrame({"id": ids, "segment": [f"seg_{value}" for value in ids]})

    enricher = AsyncFrameEnricher(
        [
            AttachmentSpec(
                key="customer_segment",
                required_cols={"customer_id"},
                attachment_fn=attachment_fn,
                col_to_kwarg={"customer_id": "ids"},
                left_on=["customer_id"],
                right_on=["id"],
                drop_cols=["id"],
            )
        ]
    )

    out = await enricher.aenrich(base_frame, cols=["customer_segment"])
    computed = out.compute().sort_values("customer_id").reset_index(drop=True)
    assert computed["segment"].tolist() == ["seg_1", "seg_2", "seg_3"]


@pytest.mark.asyncio
async def test_async_frame_enricher_respects_requested_keys(base_frame):
    async def attachment_fn(ids):
        return pd.DataFrame({"id": ids, "segment": [f"seg_{value}" for value in ids]})

    enricher = AsyncFrameEnricher(
        [
            AttachmentSpec(
                key="customer_segment",
                required_cols={"customer_id"},
                attachment_fn=attachment_fn,
                col_to_kwarg={"customer_id": "ids"},
                left_on=["customer_id"],
                right_on=["id"],
                drop_cols=["id"],
            )
        ]
    )

    out = await enricher.aenrich(base_frame, cols=["other_spec"])
    computed = out.compute()
    assert "segment" not in computed.columns


@pytest.mark.asyncio
async def test_async_frame_enricher_enforces_unique_limits(base_frame):
    async def attachment_fn(ids):
        return pd.DataFrame({"id": ids, "segment": [f"seg_{value}" for value in ids]})

    enricher = AsyncFrameEnricher(
        [
            AttachmentSpec(
                key="customer_segment",
                required_cols={"customer_id"},
                attachment_fn=attachment_fn,
                col_to_kwarg={"customer_id": "ids"},
                left_on=["customer_id"],
                right_on=["id"],
                drop_cols=["id"],
                max_unique_values=2,
            )
        ]
    )

    with pytest.raises(ValueError, match="exceeded unique-value limit"):
        await enricher.aenrich(base_frame, cols=["customer_segment"])
