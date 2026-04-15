"""
SQLAlchemy Model Builder mapping declarative ORM constructs towards the Thread-Safe Registry.
"""
from __future__ import annotations

import keyword
import re
from typing import Type, Any, Optional, Union

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from pydantic import BaseModel, Field, ConfigDict, field_validator
from boti.core.security import is_valid_dotted_identifier
from boti_data.db.sql_model_registry import get_global_registry


class BuilderConfig(BaseModel):
    """Structure managing contextual format overrides across the model boundaries."""
    model_config = ConfigDict(frozen=True)
    
    module_label: Optional[str] = Field(default=None)
    prefer_stable_names: bool = Field(default=True)

    @field_validator("module_label")
    @classmethod
    def validate_module_label(cls, value: Optional[str]) -> Optional[str]:
        """Restrict runtime module injection targets to valid module paths."""
        if value is None:
            return None
        if not is_valid_dotted_identifier(value):
            raise ValueError("module_label must be a valid dotted Python module path.")
        return value


class SqlAlchemyModelBuilder:
    """
    Builds a single SQLAlchemy ORM model representing a target database table.
    Delegates dynamic reflection tracking natively to the caching registry.
    """

    def __init__(self, engine: Union[Engine, AsyncEngine], table_name: str, config: Optional[BuilderConfig] = None):
        self.engine = engine
        self.table_name = table_name
        self.config = config or BuilderConfig()

    def build_model(self) -> Type[Any]:
        """Reflects the table and returns the stable mapped ORM class."""
        registry = get_global_registry()

        return registry.get_model(
            engine=self.engine,
            table_name=self.table_name,
            module_label=self.config.module_label,
            prefer_stable_names=self.config.prefer_stable_names,
        )

    async def build_model_async(self) -> Type[Any]:
        """Reflects the table and returns the stable mapped ORM class across parallel async loops."""
        registry = get_global_registry()

        return await registry.get_model_async(
            engine=self.engine,  # type: ignore
            table_name=self.table_name,
            module_label=self.config.module_label,
            prefer_stable_names=self.config.prefer_stable_names,
        )

    @staticmethod
    def _normalize_class_name(table_name: str) -> str:
        return "".join(word.capitalize() for word in table_name.split("_"))

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        # Prevents mapping columns matching reserved python syntax
        sane_name = re.sub(r"\W", "_", column_name)
        sane_name = re.sub(r"^\d", r"_\g<0>", sane_name)
        if keyword.iskeyword(sane_name):
            return f"{sane_name}_field"
        return sane_name
