from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from boti_data.gateway.requests import ExecutionMode, ReturnType


class GatewayLoadRequest(BaseModel):
    """Typed load request for the gateway layer.

    Captures every field in ``LOAD_CONTROL_KEYS`` as a typed attribute,
    replacing ``dict[str, Any]`` option passing across architectural boundaries.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Source
    sql: str | None = None
    statement: Any = None
    model: Any = None
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    # Projection / limit
    limit: int | None = Field(default=None, ge=0)
    columns: list[str] | None = None

    # Output type
    return_type: ReturnType | None = None
    execution_mode: ExecutionMode | None = None
    as_pandas: bool = False

    # Execution controls
    persist: bool = False
    resilient: bool = False
    diagnostics: bool = False
    timeout: float | None = None
    dry_run: bool = False
    allow_raw_sql: bool = False

    # IN-chunking controls
    in_chunk_size: int | None = None
    in_chunk_concurrency: int | None = None
    in_chunk_strategy: str = "auto"

    # Partitioned load controls
    partitioned: bool | None = None
    partition_strategy: str | None = None
    partition_column: str | None = None
    order_column: str | None = None
    chunk_size: int | None = Field(default=None, ge=1)
    max_concurrent_fetches: int | None = None

    # Backend-specific
    raw_filters: list[Any] | None = None
    cube: Any = None

    @model_validator(mode="after")
    def _validate_source(self) -> GatewayLoadRequest:
        if self.sql is not None and self.statement is not None:
            raise ValueError("Provide either sql or statement, not both.")
        if self.statement is not None and self.model is None:
            raise ValueError("statement requires model.")
        return self

    def is_configured(self, gateway_configured: bool) -> bool:
        """Return ``True`` when this request is for a configured-mode gateway.

        Deferred to the gateway's own configured flag since the request
        itself cannot know whether the gateway owns a fixed *table*.
        """
        return gateway_configured

    def build_loader_options(self) -> dict[str, Any]:
        """Return a dict of structured fields for backend loader dispatch.

        Excludes IN-chunking and execution controls that are consumed
        upstream by :class:`~boti_data.gateway.planning.LoadPlanner`.
        """
        return {
            "sql": self.sql,
            "statement": self.statement,
            "model": self.model,
            "filters": self.filters,
            "params": self.params,
            "limit": self.limit,
            "columns": self.columns,
            "as_pandas": self.as_pandas,
            "diagnostics": self.diagnostics,
        }


__all__ = ["GatewayLoadRequest"]
