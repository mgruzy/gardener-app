#!/usr/bin/env python
"""Entry point for the plant_db_team crew."""

import json
import os
import warnings
import urllib.request
from pathlib import Path

import yaml

from plant_db_team.crew import PlantDbTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

PLANT_LIST_PATH = Path(__file__).parent / "config" / "plant_list.yaml"
GARDEN_DATA_DIR = Path(__file__).parents[4] / "data"

DEFAULT_DB_PATH = str(GARDEN_DATA_DIR / "plant.duckdb")
DEFAULT_ZIPCODE = "98115"  # seattle
BATCH_SIZE = 10


def lookup_hardiness_zone(zipcode: str) -> str:
    """
    Look up the USDA plant hardiness zone for a given US zipcode.

    Uses the public PHZM API (phzmapi.org). Returns "9a" as fallback
    if the lookup fails or the zipcode is not found.
    """
    url = f"https://phzmapi.org/{zipcode}.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            return data.get("zone", "9a")
    except Exception:
        return "9a"


def build_batches(path: Path, batch_size: int) -> list[tuple[str, str]]:
    """
    Load plant_list.yaml and split into batches of batch_size plants.

    Args:
        path: Path to plant_list.yaml.
        batch_size: Number of plants per crew run.

    Returns:
        List of (plant_categories, plant_list) tuples — one per batch.
        plant_categories is always the full category set (used for enum validation).
        plant_list contains only the plants for the current batch.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    all_categories = list(data.keys())
    plant_categories = ", ".join(all_categories)

    # flatten to (category, name) pairs preserving order
    flat: list[tuple[str, str]] = []
    for category, plants in data.items():
        for plant in plants:
            flat.append((category, plant))

    batches: list[tuple[str, str]] = []
    for i in range(0, len(flat), batch_size):
        chunk = flat[i : i + batch_size]

        by_category: dict[str, list[str]] = {}
        for category, plant in chunk:
            by_category.setdefault(category, []).append(plant)

        lines: list[str] = []
        for category, plants in by_category.items():
            lines.append(f"{category.upper()}:")
            lines.extend(f"  - {p}" for p in plants)

        batches.append((plant_categories, "\n".join(lines)))

    return batches


def run() -> None:
    """Run the plant database builder crew in batches."""
    os.makedirs("output", exist_ok=True)
    GARDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    batches = build_batches(PLANT_LIST_PATH, BATCH_SIZE)
    zipcode = os.environ.get("GARDEN_ZIPCODE", DEFAULT_ZIPCODE)
    hardiness_zone = lookup_hardiness_zone(zipcode)
    db_path = os.environ.get("PLANT_DB_PATH", DEFAULT_DB_PATH)

    total = len(batches)
    print(f"\nStarting plant DB build — {total} batches of up to {BATCH_SIZE} plants each\n")

    for i, (plant_categories, plant_list) in enumerate(batches, 1):
        plant_count = plant_list.count("  - ")
        print(f"--- Batch {i}/{total} ({plant_count} plants) ---")
        inputs = {
            "plant_categories": plant_categories,
            "plant_list": plant_list,
            "zipcode": zipcode,
            "hardiness_zone": hardiness_zone,
            "db_path": db_path,
        }
        try:
            PlantDbTeam().crew().kickoff(inputs=inputs)
        except Exception as e:
            raise Exception(f"Batch {i}/{total} failed: {e}") from e

    print(f"\nAll {total} batches complete.")


if __name__ == "__main__":
    run()
