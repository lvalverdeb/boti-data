"""Config/type definitions for SqlModelRegistry.

Split out of sql_model_registry.py purely for line-count headroom — these
types have no behavioral dependency on SqlModelRegistry itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boti.core.security import is_valid_dotted_identifier
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import DeclarativeBase


class RegistryConfig(BaseModel):
    """Configuration mapping establishing deterministic registry footprints."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    default_module_label: str = Field(default="boti_data.dynamic_models")

    @field_validator("default_module_label")
    @classmethod
    def validate_default_module_label(cls, value: str) -> str:
        """Restrict registry module targets to valid module paths."""
        if not is_valid_dotted_identifier(value):
            raise ValueError("default_module_label must be a valid dotted Python module path.")
        return value


class DefaultBase(DeclarativeBase):
    """Shared declarative base for prototype ORM models."""

    pass


@dataclass(frozen=True)
class ModelBuildContext:
    """Table-independent context shared by get_model()/get_model_async() when
    calling _build_model_class()."""

    qname: str
    tname: str
    base_class: type[Any]
    module_label: str
    prefer_stable_names: bool
    ekey: str
    schema: str | None
    missing_pk_warning_suffix: str
