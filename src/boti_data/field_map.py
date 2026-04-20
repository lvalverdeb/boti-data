"""
Bidirectional translator between raw DB column names and clean semantic names.

field_map convention:  {db_column_name: semantic_name}
  e.g. {'id_tipo_producto': 'product_type_id'}
"""
from __future__ import annotations

from typing import Any, Literal

InputKeyMode = Literal["db", "semantic"]


class FieldMap:
    """
    Translates filter keys, selects columns, and renames DataFrame output
    between raw database column names and application-level semantic names.

    The mapping direction is always  db_column -> semantic_name  (same as
    Django-style field_map / DfHelper convention).
    """

    def __init__(self, mapping: dict[str, str], *, strict: bool = False) -> None:
        self._db_to_semantic: dict[str, str] = dict(mapping)
        self._strict: bool = strict
        # Build reverse lookup once; duplicate semantic values are an error
        seen: dict[str, str] = {}
        for db_col, sem in mapping.items():
            if sem in seen:
                raise ValueError(
                    f"Duplicate semantic name '{sem}' maps to both "
                    f"'{seen[sem]}' and '{db_col}'."
                )
            seen[sem] = db_col
        self._semantic_to_db: dict[str, str] = {v: k for k, v in mapping.items()}

    def __bool__(self) -> bool:
        return bool(self._db_to_semantic)

    def __repr__(self) -> str:
        return f"FieldMap({len(self._db_to_semantic)} columns)"

    @property
    def db_columns(self) -> list[str]:
        return list(self._db_to_semantic.keys())

    @property
    def semantic_names(self) -> list[str]:
        return list(self._db_to_semantic.values())

    def to_db(self, key: str) -> str:
        """Translate a semantic name to the DB column name.

        When ``strict=True`` (set at construction), raises ``KeyError`` for
        unknown keys instead of passing them through unchanged.  This prevents
        attacker-controlled keys from leaking into downstream SQL resolution.

        Returns *key* unchanged (when ``strict=False``) for unknown keys so that
        already-DB keys pass through safely in non-strict contexts.
        """
        if key in self._semantic_to_db:
            return self._semantic_to_db[key]
        if self._strict:
            raise KeyError(
                f"Unknown semantic field '{key}': not found in FieldMap. "
                "Pass strict=False or add the field to the mapping."
            )
        return key

    def to_semantic(self, key: str) -> str:
        """Translate a DB column name to its semantic name.

        Returns *key* unchanged when the column has no entry in the map.
        """
        return self._db_to_semantic.get(key, key)

    # ------------------------------------------------------------------
    # Filter translation
    # ------------------------------------------------------------------

    def translate_filters_to_db(
        self,
        filters: dict[str, Any],
        *,
        input_keys_are: InputKeyMode = "semantic",
    ) -> dict[str, Any]:
        """Return a copy of *filters* with all field keys translated to DB column names.

        Handles:
        - Nested ``$and`` / ``$or`` / ``$not`` operators (recursively)
        - ``field__op`` and ``field__casting__op`` key syntax — only the
          leading field segment is translated; operator suffixes are preserved

        When ``input_keys_are="db"`` the dict is returned as-is (no translation
        needed; the keys are already DB column names).
        """
        if input_keys_are == "db" or not self._semantic_to_db:
            return dict(filters)

        result: dict[str, Any] = {}
        for key, value in filters.items():
            if key.startswith("$"):
                # Boolean operator — recurse into sub-filters
                if isinstance(value, list):
                    result[key] = [
                        self.translate_filters_to_db(sub, input_keys_are=input_keys_are)
                        if isinstance(sub, dict)
                        else sub
                        for sub in value
                    ]
                elif isinstance(value, dict):
                    result[key] = self.translate_filters_to_db(value, input_keys_are=input_keys_are)
                else:
                    result[key] = value
                continue

            # Translate the leading field segment; preserve __op suffixes
            parts = key.split("__")
            parts[0] = self.to_db(parts[0])
            result["__".join(parts)] = value

        return result

    # ------------------------------------------------------------------
    # Column selection helpers
    # ------------------------------------------------------------------

    def select_db_columns(self, semantic_names: list[str] | tuple[str, ...]) -> list[str]:
        """Convert a list/tuple of semantic column names to DB column names."""
        return [self.to_db(name) for name in semantic_names]

    # ------------------------------------------------------------------
    # DataFrame renaming
    # ------------------------------------------------------------------

    def rename_dataframe(self, df: Any) -> Any:
        """Rename DB column names to semantic names in the output DataFrame.

        Only columns present in *df* are renamed; extras in the map are ignored.
        Works with both pandas and Dask DataFrames.
        """
        rename_map = {k: v for k, v in self._db_to_semantic.items() if k in df.columns}
        if rename_map:
            return df.rename(columns=rename_map)
        return df

    def rename_table(self, table: Any) -> Any:
        """Rename DB column names to semantic names in a PyArrow Table.

        Uses ``table.rename_columns()`` which is O(1) — no data movement.
        Only columns present in *table* are renamed.
        """
        import pyarrow as pa

        if not isinstance(table, pa.Table):
            raise TypeError(f"Expected pyarrow.Table, got {type(table).__name__}")

        # Build rename map: only for columns that exist in the table
        current_names = table.column_names
        rename_pairs = []
        for db_col, sem_col in self._db_to_semantic.items():
            if db_col in current_names:
                rename_pairs.append((db_col, sem_col))

        if not rename_pairs:
            return table

        # Build new column names list
        new_names = list(current_names)
        for db_col, sem_col in rename_pairs:
            idx = new_names.index(db_col)
            new_names[idx] = sem_col

        return table.rename_columns(new_names)

    def apply_column_names(self, df: Any, column_names: list[str]) -> Any:
        """Positionally rename all output columns via an explicit list.

        Useful when the consumer wants names that differ from both the DB
        and semantic conventions.  ``column_names`` must cover every column
        that is currently in *df*.
        """
        current = list(df.columns)
        if len(column_names) != len(current):
            raise ValueError(
                f"column_names length ({len(column_names)}) does not match "
                f"DataFrame column count ({len(current)}): {current}."
            )
        return df.rename(columns=dict(zip(current, column_names)))
