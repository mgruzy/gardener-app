"""
Schema initialization for garden.duckdb, weather.duckdb, and chat_memory.duckdb.
plant.duckdb schema is owned entirely by the CrewAI plant_db_team agent (Phase 2).
Call initialize_all_schemas() once at app startup.
"""

import duckdb

from app.db.connections import DatabaseConnections


def build_garden_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS garden (
            garden_id       VARCHAR PRIMARY KEY,
            name            VARCHAR NOT NULL,
            zipcode         VARCHAR NOT NULL,
            aerial_image_path VARCHAR,
            lat             DOUBLE,
            lon             DOUBLE,
            created_at      DATE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS plots (
            plot_id             VARCHAR PRIMARY KEY,
            garden_id           VARCHAR NOT NULL,
            name                VARCHAR NOT NULL,
            polygon             JSON NOT NULL,
            area_sqft           DOUBLE,
            sun_zone_default    VARCHAR,
            height_zone_default VARCHAR,
            sun_zone_regions    JSON,
            height_zone_regions JSON,
            notes               VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS plant_instances (
            instance_id         VARCHAR PRIMARY KEY,
            plot_id             VARCHAR NOT NULL,
            plant_id            VARCHAR NOT NULL,
            plant_type_id       VARCHAR NOT NULL,
            location_x          DOUBLE,
            location_y          DOUBLE,
            status              VARCHAR NOT NULL,
            planned_sow_date    DATE,
            planted_date        DATE,
            removed_date        DATE,
            harvest_count       INTEGER DEFAULT 0,
            harvest_weight_lbs  DOUBLE DEFAULT 0.0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS plant_photos (
            photo_id            VARCHAR PRIMARY KEY,
            plant_instance_id   VARCHAR NOT NULL,
            plant_type_id       VARCHAR NOT NULL,
            plant_id            VARCHAR NOT NULL,
            plant_type          VARCHAR NOT NULL,
            name_of_file        VARCHAR NOT NULL,
            date_added          DATE NOT NULL,
            notes               VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS garden_snapshots (
            snapshot_id     VARCHAR PRIMARY KEY,
            snapshot_date   DATE NOT NULL,
            triggered_by    VARCHAR NOT NULL,
            snapshot_json   JSON NOT NULL,
            is_current      BOOLEAN DEFAULT FALSE
        )
    """)


def build_weather_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_records (
            zipcode             VARCHAR NOT NULL,
            date                DATE NOT NULL,
            temp_min_f          DOUBLE,
            temp_max_f          DOUBLE,
            precipitation_in    DOUBLE,
            humidity_pct        DOUBLE,
            pressure_inhg       DOUBLE,
            wind_min_mph        DOUBLE,
            wind_max_mph        DOUBLE,
            wind_direction      VARCHAR,
            air_quality_index   DOUBLE,
            PRIMARY KEY (zipcode, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_metadata (
            zipcode             VARCHAR PRIMARY KEY,
            first_record_date   DATE,
            last_updated_date   DATE
        )
    """)


def build_chat_memory_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id       VARCHAR PRIMARY KEY,
            created_at      TIMESTAMP NOT NULL,
            topic_summary   VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id  VARCHAR PRIMARY KEY,
            thread_id   VARCHAR NOT NULL,
            role        VARCHAR NOT NULL,
            content     VARCHAR NOT NULL,
            timestamp   TIMESTAMP NOT NULL
        )
    """)


def initialize_all_schemas(connections: DatabaseConnections) -> None:
    """Create all tables across garden, weather, and chat_memory databases."""
    build_garden_db(connections.garden)
    build_weather_db(connections.weather)
    build_chat_memory_db(connections.chat_memory)
