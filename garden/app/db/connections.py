"""
DatabaseConnections — returns DuckDB connections for all four databases.
In LIVE mode: connects to real .duckdb files.
In TEST mode: copies real data into :memory: so nothing is written back to disk.
"""

import os
from dataclasses import dataclass

import duckdb

from app.db.config import AppConfig


@dataclass
class DatabaseConnections:
    """Holds open DuckDB connections for all four databases."""

    garden: duckdb.DuckDBPyConnection
    weather: duckdb.DuckDBPyConnection
    chat_memory: duckdb.DuckDBPyConnection
    plant: duckdb.DuckDBPyConnection

    def close_all(self) -> None:
        """Close all open connections."""
        for conn in [self.garden, self.weather, self.chat_memory, self.plant]:
            conn.close()


def copy_to_memory(source_path: str) -> duckdb.DuckDBPyConnection:
    """
    Copy all tables from a real DuckDB file into a fresh :memory: connection.
    If the source file does not exist yet, returns an empty memory connection.
    """
    mem_conn = duckdb.connect(":memory:")
    if not os.path.exists(source_path):
        return mem_conn

    mem_conn.execute(f"ATTACH '{source_path}' AS src (READ_ONLY)")
    tables = mem_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'src'"
    ).fetchall()
    for (table_name,) in tables:
        mem_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM src.{table_name}")
    mem_conn.execute("DETACH src")
    return mem_conn


def open_connections(config: AppConfig) -> DatabaseConnections:
    """
    Open connections to all four databases based on the app config mode.
    LIVE  → real .duckdb files (reads + writes persisted)
    TEST  → :memory: copies of real files (nothing written back to disk)
    """
    if config.is_test:
        return DatabaseConnections(
            garden=copy_to_memory(config.garden_db_path),
            weather=copy_to_memory(config.weather_db_path),
            chat_memory=copy_to_memory(config.chat_memory_db_path),
            plant=copy_to_memory(config.plant_db_path),
        )

    return DatabaseConnections(
        garden=duckdb.connect(config.garden_db_path),
        weather=duckdb.connect(config.weather_db_path),
        chat_memory=duckdb.connect(config.chat_memory_db_path),
        plant=duckdb.connect(config.plant_db_path),
    )
