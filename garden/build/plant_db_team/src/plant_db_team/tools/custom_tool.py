"""PlantDatabaseWriter — CrewAI tool that persists validated plant data to plant.duckdb."""

import json
from typing import Type
from uuid import uuid4

import duckdb
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_CREATE_PLANT_TYPES = """
    CREATE TABLE IF NOT EXISTS plant_types (
        plant_type_id   VARCHAR PRIMARY KEY,
        name            VARCHAR NOT NULL UNIQUE,
        category        VARCHAR NOT NULL
    )
"""

_CREATE_PLANT_VARIETIES = """
    CREATE TABLE IF NOT EXISTS plant_varieties (
        variety_id                      VARCHAR PRIMARY KEY,
        plant_type_id                   VARCHAR NOT NULL,
        variety_name                    VARCHAR NOT NULL,
        plant_category                  VARCHAR NOT NULL,
        sun_tolerance                   VARCHAR NOT NULL,
        water_required                  VARCHAR NOT NULL,
        soil_n                          DOUBLE,
        soil_p                          DOUBLE,
        soil_k                          DOUBLE,
        growth_needs                    VARCHAR,
        post_harvest_soil_needs         VARCHAR,
        days_to_harvest                 INTEGER,
        indoor_sow_weeks_before_frost   INTEGER,
        outdoor_sow_date_range          VARCHAR,
        spacing_inches                  DOUBLE,
        harvest_timing                  VARCHAR,
        temp_min_air_f                  DOUBLE,
        temp_min_ground_f               DOUBLE,
        height_inches_estimate          DOUBLE
    )
"""

_CREATE_PLANT_COMPANIONS = """
    CREATE TABLE IF NOT EXISTS plant_companions (
        companion_id    VARCHAR PRIMARY KEY,
        plant_type_id   VARCHAR NOT NULL,
        companion_name  VARCHAR NOT NULL,
        relationship    VARCHAR NOT NULL
    )
"""

_CREATE_PLANT_PESTS = """
    CREATE TABLE IF NOT EXISTS plant_pests (
        pest_id         VARCHAR PRIMARY KEY,
        plant_type_id   VARCHAR NOT NULL,
        pest_name       VARCHAR NOT NULL,
        symptoms        VARCHAR,
        treatment       VARCHAR
    )
"""

_CREATE_PLANT_DISEASES = """
    CREATE TABLE IF NOT EXISTS plant_diseases (
        disease_id      VARCHAR PRIMARY KEY,
        plant_type_id   VARCHAR NOT NULL,
        disease_name    VARCHAR NOT NULL,
        symptoms        VARCHAR,
        treatment       VARCHAR
    )
"""


class PlantDatabaseWriterInput(BaseModel):
    """Input schema for PlantDatabaseWriter."""

    plant_data_json: str = Field(
        ...,
        description="The full validated JSON array of plant objects to write to the database.",
    )
    db_path: str = Field(
        ...,
        description="Absolute path to the plant.duckdb file.",
    )


class PlantDatabaseWriter(BaseTool):
    name: str = "PlantDatabaseWriter"
    description: str = (
        "Write a validated JSON array of plant data to plant.duckdb. "
        "Creates all required tables if they do not exist. "
        "Skips any plant_type that already exists in the database to avoid duplicates. "
        "Returns a row-count report for each table written."
    )
    args_schema: Type[BaseModel] = PlantDatabaseWriterInput

    def _run(self, plant_data_json: str, db_path: str) -> str:
        try:
            plants = json.loads(plant_data_json)
        except json.JSONDecodeError as e:
            return f"ERROR: Failed to parse plant_data_json — {e}"

        if not isinstance(plants, list):
            return "ERROR: plant_data_json must be a JSON array."

        conn = duckdb.connect(db_path)
        conn.execute(_CREATE_PLANT_TYPES)
        conn.execute(_CREATE_PLANT_VARIETIES)
        conn.execute(_CREATE_PLANT_COMPANIONS)
        conn.execute(_CREATE_PLANT_PESTS)
        conn.execute(_CREATE_PLANT_DISEASES)

        counts: dict[str, int] = {
            "plant_types": 0,
            "plant_varieties": 0,
            "plant_companions": 0,
            "plant_pests": 0,
            "plant_diseases": 0,
        }
        skipped = 0

        for plant in plants:
            pt = plant.get("plant_type", {})
            plant_name = str(pt.get("name", "")).strip()
            if not plant_name:
                continue

            existing = conn.execute(
                "SELECT plant_type_id FROM plant_types WHERE name = ?",
                [plant_name],
            ).fetchone()

            if existing:
                skipped += 1
                continue

            plant_type_id = str(uuid4())
            conn.execute(
                "INSERT INTO plant_types (plant_type_id, name, category) VALUES (?, ?, ?)",
                [plant_type_id, plant_name, pt.get("category", "")],
            )
            counts["plant_types"] += 1

            pv = plant.get("plant_variety", {})
            conn.execute(
                """
                INSERT INTO plant_varieties (
                    variety_id, plant_type_id, variety_name, plant_category,
                    sun_tolerance, water_required, soil_n, soil_p, soil_k,
                    growth_needs, post_harvest_soil_needs, days_to_harvest,
                    indoor_sow_weeks_before_frost, outdoor_sow_date_range,
                    spacing_inches, harvest_timing, temp_min_air_f,
                    temp_min_ground_f, height_inches_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid4()),
                    plant_type_id,
                    pv.get("variety_name", "Common"),
                    pv.get("plant_category", pt.get("category", "")),
                    pv.get("sun_tolerance", "full_sun"),
                    pv.get("water_required", "medium"),
                    pv.get("soil_n"),
                    pv.get("soil_p"),
                    pv.get("soil_k"),
                    pv.get("growth_needs"),
                    pv.get("post_harvest_soil_needs"),
                    pv.get("days_to_harvest"),
                    pv.get("indoor_sow_weeks_before_frost"),
                    pv.get("outdoor_sow_date_range"),
                    pv.get("spacing_inches"),
                    pv.get("harvest_timing"),
                    pv.get("temp_min_air_f"),
                    pv.get("temp_min_ground_f"),
                    pv.get("height_inches_estimate"),
                ],
            )
            counts["plant_varieties"] += 1

            for name in plant.get("companions", []):
                conn.execute(
                    "INSERT INTO plant_companions (companion_id, plant_type_id, companion_name, relationship) VALUES (?, ?, ?, ?)",
                    [str(uuid4()), plant_type_id, name, "companion"],
                )
                counts["plant_companions"] += 1

            for name in plant.get("antagonists", []):
                conn.execute(
                    "INSERT INTO plant_companions (companion_id, plant_type_id, companion_name, relationship) VALUES (?, ?, ?, ?)",
                    [str(uuid4()), plant_type_id, name, "antagonist"],
                )
                counts["plant_companions"] += 1

            for pest in plant.get("pests", []):
                conn.execute(
                    "INSERT INTO plant_pests (pest_id, plant_type_id, pest_name, symptoms, treatment) VALUES (?, ?, ?, ?, ?)",
                    [
                        str(uuid4()),
                        plant_type_id,
                        pest.get("pest_name", ""),
                        pest.get("symptoms", ""),
                        pest.get("treatment", ""),
                    ],
                )
                counts["plant_pests"] += 1

            for disease in plant.get("diseases", []):
                conn.execute(
                    "INSERT INTO plant_diseases (disease_id, plant_type_id, disease_name, symptoms, treatment) VALUES (?, ?, ?, ?, ?)",
                    [
                        str(uuid4()),
                        plant_type_id,
                        disease.get("disease_name", ""),
                        disease.get("symptoms", ""),
                        disease.get("treatment", ""),
                    ],
                )
                counts["plant_diseases"] += 1

        conn.close()

        lines = [f"PlantDatabaseWriter — write complete", f"  Skipped (already exist): {skipped}"]
        for table, count in counts.items():
            lines.append(f"  {table}: {count} rows written")
        return "\n".join(lines)
