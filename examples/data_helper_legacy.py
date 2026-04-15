"""
DataHelper example using legacy-style configured SQL options and engine views.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper


class Base(DeclarativeBase):
    pass


class LegacyProduct(Base):
    __tablename__ = "legacy_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_tipo_produto: Mapped[int]
    codigo_barra: Mapped[str]
    nombre_th: Mapped[str]


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "legacy.db"
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                session.add_all(
                    [
                        LegacyProduct(id_tipo_produto=1, codigo_barra="A-001", nombre_th="Alice"),
                        LegacyProduct(id_tipo_produto=1, codigo_barra="A-002", nombre_th="Ana"),
                        LegacyProduct(id_tipo_produto=2, codigo_barra="B-001", nombre_th="Bob"),
                    ]
                )
                session.commit()
        finally:
            engine.dispose()

        helper = DataHelper.from_legacy_config(
            {
                "backend": "sqlalchemy",
                "connection_url": f"sqlite:///{db_path}",
                "poolclass": "sqlalchemy.pool.NullPool",
                "query_only": False,
                "table": "legacy_products",
                "field_map": {
                    "id_tipo_produto": "product_type_id",
                    "codigo_barra": "barcode",
                    "nombre_th": "first_name",
                },
                "sticky_filters": {"product_type_id": 1},
                "df_params": {"fieldnames": ("product_type_id", "barcode", "first_name")},
                "df_options": {"sort_field": "barcode"},
            }
        )
        try:
            dask_frame = helper.dask.load()
            pandas_frame = helper.pandas.load()
            polars_frame = helper.polars.load()
            dask_records = dask_frame.compute().to_dict(orient="records")
            pandas_records = pandas_frame.to_dict(orient="records")
            polars_records = polars_frame.to_dicts()
        finally:
            helper.close()

    return {
        "rows": len(pandas_records),
        "records": pandas_records,
        "dask_records": dask_records,
        "pandas_records": pandas_records,
        "polars_records": polars_records,
    }


def main() -> dict[str, object]:
    result = run_example()
    print("Legacy-config helper records:")
    for record in result["records"]:
        print(record)
    print("\nLegacy-config helper records (dask):")
    for record in result["dask_records"]:
        print(record)
    print("\nLegacy-config helper records (pandas):")
    for record in result["pandas_records"]:
        print(record)
    print("\nLegacy-config helper records (polars):")
    for record in result["polars_records"]:
        print(record)
    return result


if __name__ == "__main__":
    main()
