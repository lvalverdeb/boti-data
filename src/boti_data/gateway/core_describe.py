"""DataGateway.describe()/adescribe() implementations.

Split out of core.py purely for line-count headroom, mirroring core_load.py's
split: DataGateway's public methods keep their docstrings and stay callable
exactly as before, but their bodies move here as free functions taking the
gateway instance explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .requests import TableDescription

if TYPE_CHECKING:
    from .core import DataGateway


def resolve_describe_table(gateway: DataGateway, table: str | None) -> str:
    resolved = table or gateway._table
    if resolved is None:
        raise ValueError(
            "describe()/adescribe() require table= (or a table configured at construction time)."
        )
    return resolved


# Not a copy-pasted twin: both are thin delegates to
# gateway._strategy.describe()/describe_async() after the identical
# table-resolution step; the real logic lives one layer down (sql_describe.py).
# spaghetti-ignore[sync-async-duplication]: see above
def describe(gateway: DataGateway, table: str | None, *, row_count_limit: int) -> TableDescription:
    resolved_table = resolve_describe_table(gateway, table)
    return gateway._strategy.describe(
        gateway.resource, resolved_table, row_count_limit=row_count_limit
    )


async def describe_async(
    gateway: DataGateway, table: str | None, *, row_count_limit: int
) -> TableDescription:
    resolved_table = resolve_describe_table(gateway, table)
    return await gateway._strategy.describe_async(
        gateway.resource, resolved_table, row_count_limit=row_count_limit
    )
