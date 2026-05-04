"""
AppConfig — parsed at startup from CLI flags.
Controls whether the app runs against real databases or in-memory test copies.
"""

import argparse
from dataclasses import dataclass

import app.internal_objects.garden_types as garden_enum

DB_DIR = "data"

GARDEN_DB_PATH = f"{DB_DIR}/garden.duckdb"
WEATHER_DB_PATH = f"{DB_DIR}/weather.duckdb"
CHAT_MEMORY_DB_PATH = f"{DB_DIR}/chat_memory.duckdb"
PLANT_DB_PATH = f"{DB_DIR}/plant.duckdb"


@dataclass
class AppConfig:
    """
    Runtime configuration for the garden app.
    mode=LIVE  → reads and writes to real .duckdb files
    mode=TEST  → copies real data into :memory: on startup, no writes back to disk
    """

    mode: garden_enum.AppMode
    garden_db_path: str = GARDEN_DB_PATH
    weather_db_path: str = WEATHER_DB_PATH
    chat_memory_db_path: str = CHAT_MEMORY_DB_PATH
    plant_db_path: str = PLANT_DB_PATH

    @property
    def is_test(self) -> bool:
        return self.mode == garden_enum.AppMode.TEST


def load_config() -> AppConfig:
    """Parse --test flag from CLI args and return an AppConfig."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    mode = garden_enum.AppMode.TEST if args.test else garden_enum.AppMode.LIVE
    return AppConfig(mode=mode)
