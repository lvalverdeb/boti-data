"""R4: a pandas-only install never loads Dask, polars or boti-dask.

Runs in a subprocess with those packages made unimportable, then performs the
loads a pandas user would: a multi-partition SQL load to pandas, a Parquet
load to pandas, and a streaming read. Any import of a blocked package fails
the subprocess, so the test proves the extras split rather than assuming it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sqlite3, sys

    BLOCKED = {"dask", "distributed", "boti_dask", "polars"}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Blocker())
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]

    import boti_data
    from boti_data import DataGateway, SqlRepository
    from boti_data.db.sql_config import SqlDatabaseConfig

    db, pq = sys.argv[1], sys.argv[2]
    gateway = DataGateway(SqlDatabaseConfig(connection_url=f"sqlite:///{db}", query_only=False), table="t")
    with gateway:
        frame = gateway.load(as_pandas=True, return_type="pandas", chunk_size=100)
        streamed = sum(b.num_rows for b in gateway.iter_batches(batch_size=50))
    assert len(frame) == 1000 and streamed == 1000, (len(frame), streamed)

    pgw = DataGateway.from_backend("parquet", parquet_storage_path=pq)
    with pgw:
        pframe = pgw.load(as_pandas=True, return_type="pandas")
    assert len(pframe) == 20, len(pframe)

    with SqlRepository(SqlDatabaseConfig(connection_url=f"sqlite:///{db}"), "t", query_only=False) as repo:
        assert repo.get(1)["v"] == 1.0

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("pandas-only ok")
    """
)


@pytest.fixture
def sources(tmp_path) -> tuple[str, str]:
    db = tmp_path / "t.sqlite"
    con = sqlite3_connect(db)
    con.execute("create table t (id integer primary key, v real)")
    con.executemany("insert into t values (?, ?)", ((i, float(i)) for i in range(1, 1001)))
    con.commit()
    con.close()
    pq = tmp_path / "pq"
    pq.mkdir()
    pd.DataFrame({"id": range(20), "v": [float(i) for i in range(20)]}).to_parquet(
        pq / "a.parquet", index=False
    )
    return str(db), str(pq)


def sqlite3_connect(path):
    import sqlite3

    return sqlite3.connect(path)


def test_pandas_loads_work_with_dask_and_polars_unimportable(sources) -> None:
    db, pq = sources
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, db, pq], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "pandas-only ok" in proc.stdout


def test_missing_engine_error_names_the_extra(monkeypatch) -> None:
    from boti_data._optional import engine_required

    with pytest.raises(ModuleNotFoundError, match=r"boti-data\[dask\]"):
        with engine_required("dask"):
            raise ModuleNotFoundError("No module named 'distributed'", name="distributed")
    with pytest.raises(ModuleNotFoundError, match="not_an_engine"):
        with engine_required("dask"):
            raise ModuleNotFoundError("No module named 'not_an_engine'", name="not_an_engine")
