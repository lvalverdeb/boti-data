"""
SqlDatabaseConfig example using pydantic-settings dotenv loading.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from boti_data.db import SqlDatabaseConfig


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        env_file = Path(tmp_dir) / ".env"
        env_file.write_text(
            "DB_CONNECTION_URL='sqlite:///:memory:'\nDB_QUERY_ONLY=false\nDB_POOL_SIZE=9\n",
            encoding="utf-8",
        )

        config = SqlDatabaseConfig.from_env(env_file=env_file)

        print(config.connection_url.get_secret_value())
        print(f"query_only={config.query_only}")
        print(f"pool_size={config.pool_size}")


if __name__ == "__main__":
    main()
