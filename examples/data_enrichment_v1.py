"""AsyncFrameEnricher example: attach lookup data declaratively before downstream writes."""

from __future__ import annotations

import asyncio

import pandas as pd

from boti_data import AsyncFrameEnricher, AttachmentSpec


async def customer_segment_attachment(ids: list[int]) -> pd.DataFrame:
    # Simulate a remote lookup/loader.
    return pd.DataFrame({"id": ids, "segment": [f"seg_{value}" for value in ids]})


async def main() -> dict[str, object]:
    base = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
        }
    )

    enricher = AsyncFrameEnricher(
        [
            AttachmentSpec(
                key="customer_segment",
                required_cols={"customer_id"},
                attachment_fn=customer_segment_attachment,
                col_to_kwarg={"customer_id": "ids"},
                left_on=["customer_id"],
                right_on=["id"],
                drop_cols=["id"],
            )
        ]
    )

    enriched = await enricher.aenrich(base, cols=["customer_segment"])
    out = enriched.compute().sort_values("customer_id").reset_index(drop=True)
    summary = {
        "rows": int(len(out)),
        "segments": out["segment"].tolist(),
    }
    print(f"AsyncFrameEnricher rows={summary['rows']} segments={summary['segments']}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
