"""
Snapshot management — save, list, and restore point-in-time garden states.
Snapshots are stored in the garden_snapshots table in garden.duckdb.
"""

import json
from dataclasses import asdict
from datetime import date
from uuid import uuid4

import duckdb

import app.internal_objects.garden_types as garden_enum
from app.internal_objects.garden_objects import Garden, GardenSnapshot


def save_snapshot(
    conn: duckdb.DuckDBPyConnection,
    garden: Garden,
    trigger: garden_enum.SnapshotTrigger,
) -> str:
    """
    Serialize the full garden tree and save as a new snapshot.
    Marks all previous snapshots as is_current=False before saving.
    Returns the new snapshot_id.
    """
    snapshot_id = str(uuid4())
    snapshot_json = json.dumps(asdict(garden), default=str)

    conn.execute("UPDATE garden_snapshots SET is_current = FALSE")
    conn.execute(
        """
        INSERT INTO garden_snapshots
            (snapshot_id, snapshot_date, triggered_by, snapshot_json, is_current)
        VALUES (?, ?, ?, ?, TRUE)
        """,
        [snapshot_id, date.today(), trigger.value, snapshot_json],
    )
    return snapshot_id


def list_snapshots(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """
    Return all snapshots ordered by date descending.
    Each dict has: snapshot_id, snapshot_date, triggered_by, is_current.
    """
    rows = conn.execute(
        """
        SELECT snapshot_id, snapshot_date, triggered_by, is_current
        FROM garden_snapshots
        ORDER BY snapshot_date DESC
        """
    ).fetchall()

    return [
        {
            "snapshot_id": row[0],
            "snapshot_date": row[1],
            "triggered_by": row[2],
            "is_current": row[3],
        }
        for row in rows
    ]


def restore_snapshot(
    conn: duckdb.DuckDBPyConnection,
    snapshot_id: str,
) -> dict:
    """
    Restore the garden state from a specific snapshot.
    Overwrites plots, plant_instances, and plant_photos with snapshot data.
    Marks the restored snapshot as is_current=True.
    Returns the deserialized garden state as a dict.
    """
    row = conn.execute(
        "SELECT snapshot_json FROM garden_snapshots WHERE snapshot_id = ?",
        [snapshot_id],
    ).fetchone()

    if row is None:
        raise ValueError(f"Snapshot {snapshot_id} not found")

    garden_state = json.loads(row[0])

    # clear current state and restore from snapshot
    conn.execute("DELETE FROM plant_instances")
    conn.execute("DELETE FROM plots")
    conn.execute("DELETE FROM garden")

    garden_data = garden_state
    conn.execute(
        """
        INSERT INTO garden
            (garden_id, name, zipcode, aerial_image_path, lat, lon, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            garden_data["garden_id"],
            garden_data["name"],
            garden_data["zipcode"],
            garden_data["aerial_image_path"],
            garden_data["lat"],
            garden_data["lon"],
            garden_data["created_at"],
        ],
    )

    for plot in garden_data.get("plots", []):
        conn.execute(
            """
            INSERT INTO plots
                (plot_id, garden_id, name, polygon, area_sqft,
                 sun_zone_default, height_zone_default,
                 sun_zone_regions, height_zone_regions, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                plot["plot_id"],
                plot["garden_id"],
                plot["name"],
                json.dumps(plot["polygon"]),
                plot["area_sqft"],
                plot["sun_zone_default"],
                plot["height_zone_default"],
                json.dumps(plot["sun_zone_regions"]),
                json.dumps(plot["height_zone_regions"]),
                plot["notes"],
            ],
        )

        for instance in plot.get("plant_instances", []):
            conn.execute(
                """
                INSERT INTO plant_instances
                    (instance_id, plot_id, plant_id, plant_type_id,
                     location_x, location_y, status,
                     planned_sow_date, planted_date, removed_date,
                     harvest_count, harvest_weight_lbs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    instance["instance_id"],
                    instance["plot_id"],
                    instance["plant_id"],
                    instance["plant_type_id"],
                    instance["location_x"],
                    instance["location_y"],
                    instance["status"],
                    instance.get("planned_sow_date"),
                    instance.get("planted_date"),
                    instance.get("removed_date"),
                    instance["harvest_count"],
                    instance["harvest_weight_lbs"],
                ],
            )

    conn.execute(
        "UPDATE garden_snapshots SET is_current = FALSE"
    )
    conn.execute(
        "UPDATE garden_snapshots SET is_current = TRUE WHERE snapshot_id = ?",
        [snapshot_id],
    )

    return garden_state
