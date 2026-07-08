from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .normalization import DEFAULT_IN_CHUNK_SIZE
from .planning import InChunkPolicy


@dataclass(frozen=True)
class GatewayPolicies:
    in_chunk_policy: InChunkPolicy = field(
        default_factory=lambda: InChunkPolicy(DEFAULT_IN_CHUNK_SIZE)
    )
    chunk_hint_fn: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
