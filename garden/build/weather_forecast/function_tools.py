"""Function tools for the weather forecast agent."""

import json
import urllib.request
from datetime import date, datetime, timedelta, timezone

import duckdb
from agents import function_tool
from openai import OpenAI


# ── Parameters ────────────────────────────────────────────────────────────────

DAYS_BACK = 4
DAYS_AHEAD = 3
MODEL_SUMMARY = "gpt-4o-mini"
FROST_RISK_THRESHOLD_F = 36.0
HEAT_STRESS_THRESHOLD_F = 85.0
HIGH_WIND_THRESHOLD_MPH = 15.0
HEAVY_RAIN_THRESHOLD_IN = 0.5

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,wind_direction_10m_dominant"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=auto"
)
FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,wind_direction_10m_dominant"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&forecast_days={days}"
    "&timezone=auto"
)

CREATE_FORECAST_TABLE = """
    CREATE TABLE IF NOT EXISTS weather_forecast (
        zipcode          VARCHAR NOT NULL,
        date             DATE NOT NULL,
        record_type      VARCHAR NOT NULL,
        temp_min_f       DOUBLE,
        temp_max_f       DOUBLE,
        precipitation_in DOUBLE,
        wind_max_mph     DOUBLE,
        wind_direction   VARCHAR,
        fetched_at       TIMESTAMP,
        PRIMARY KEY (zipcode, date)
    )
"""
CREATE_SUMMARY_TABLE = """
    CREATE TABLE IF NOT EXISTS weather_forecast_summary (
        zipcode      VARCHAR PRIMARY KEY,
        summary_text VARCHAR,
        fetched_at   TIMESTAMP
    )
"""


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool
def fetch_recent_and_forecast(zipcode: str, lat: float, lon: float, weather_db_path: str) -> dict:
    """
    Fetch last 4 days of actual weather and today + 3-day forecast from Open-Meteo.

    Calls the Open-Meteo archive API for actual data and the forecast API for upcoming days.
    Writes all records to the weather_forecast table in weather.duckdb, replacing any prior
    records for this zipcode. Returns a summary of records written by type.
    """
    today = date.today()
    start_actual = today - timedelta(days=DAYS_BACK)
    end_actual = today - timedelta(days=1)

    archive_url = ARCHIVE_URL.format(lat=lat, lon=lon, start=start_actual, end=end_actual)
    forecast_url = FORECAST_URL.format(lat=lat, lon=lon, days=DAYS_AHEAD + 1)

    records: list[dict] = []

    try:
        with urllib.request.urlopen(archive_url, timeout=30) as resp:
            archive_data = json.loads(resp.read())
        daily = archive_data.get("daily", {})
        for i, d_str in enumerate(daily.get("time", [])):
            records.append({
                "date": d_str,
                "record_type": "actual",
                "temp_min_f": _safe_get(daily.get("temperature_2m_min", []), i),
                "temp_max_f": _safe_get(daily.get("temperature_2m_max", []), i),
                "precipitation_in": _safe_get(daily.get("precipitation_sum", []), i),
                "wind_max_mph": _safe_get(daily.get("wind_speed_10m_max", []), i),
                "wind_direction": _fmt_wind(_safe_get(daily.get("wind_direction_10m_dominant", []), i)),
            })
    except Exception as e:
        return {"error": f"Archive fetch failed: {e}"}

    try:
        with urllib.request.urlopen(forecast_url, timeout=30) as resp:
            forecast_data = json.loads(resp.read())
        daily = forecast_data.get("daily", {})
        for i, d_str in enumerate(daily.get("time", [])):
            records.append({
                "date": d_str,
                "record_type": "forecast",
                "temp_min_f": _safe_get(daily.get("temperature_2m_min", []), i),
                "temp_max_f": _safe_get(daily.get("temperature_2m_max", []), i),
                "precipitation_in": _safe_get(daily.get("precipitation_sum", []), i),
                "wind_max_mph": _safe_get(daily.get("wind_speed_10m_max", []), i),
                "wind_direction": _fmt_wind(_safe_get(daily.get("wind_direction_10m_dominant", []), i)),
            })
    except Exception as e:
        return {"error": f"Forecast fetch failed: {e}"}

    conn = duckdb.connect(weather_db_path)
    conn.execute(CREATE_FORECAST_TABLE)
    conn.execute("DELETE FROM weather_forecast WHERE zipcode = ?", [zipcode])
    fetched_at = datetime.now(timezone.utc)
    for r in records:
        conn.execute(
            """INSERT INTO weather_forecast
                   (zipcode, date, record_type, temp_min_f, temp_max_f,
                    precipitation_in, wind_max_mph, wind_direction, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [zipcode, r["date"], r["record_type"],
             r["temp_min_f"], r["temp_max_f"],
             r["precipitation_in"], r["wind_max_mph"],
             r["wind_direction"], fetched_at],
        )
    conn.commit()
    conn.close()

    actual_count = sum(1 for r in records if r["record_type"] == "actual")
    forecast_count = sum(1 for r in records if r["record_type"] == "forecast")
    return {
        "status": "written",
        "zipcode": zipcode,
        "actual_records": actual_count,
        "forecast_records": forecast_count,
    }


@function_tool
def generate_and_store_summary(zipcode: str, weather_db_path: str) -> dict:
    """
    Generate a gardener-focused weather briefing using OpenAI and persist it.

    Reads weather_forecast records for the zipcode, calls gpt-4o-mini to produce a 2-3 sentence
    plain-text summary noting frost risk, heavy rain, heat stress, and high wind. Stores the
    summary in weather_forecast_summary and returns the text.
    """
    conn = duckdb.connect(weather_db_path)
    rows = conn.execute(
        """SELECT date, record_type, temp_min_f, temp_max_f, precipitation_in, wind_max_mph
           FROM weather_forecast WHERE zipcode = ? ORDER BY date""",
        [zipcode],
    ).fetchall()
    frost_row = conn.execute(
        "SELECT avg_last_spring_frost, avg_first_fall_frost FROM frost_dates WHERE zipcode = ?",
        [zipcode],
    ).fetchone()
    conn.close()

    if not rows:
        return {"error": "No forecast data — run fetch_recent_and_forecast first."}

    data_lines = [
        f"{d} ({rtype}): high={tmax}°F  low={tmin}°F  precip={precip}in  wind_max={wind}mph"
        for d, rtype, tmin, tmax, precip, wind in rows
    ]
    frost_context = ""
    if frost_row:
        frost_context = (
            f"\nFrost context: avg last spring frost={frost_row[0]},"
            f" avg first fall frost={frost_row[1]}"
        )

    prompt = f"""You are a concise garden-focused weather assistant. Based on this 7-day weather window, write a 2-3 sentence briefing for a home gardener. Be direct and specific about dates.

Weather data (actual = measured, forecast = predicted):
{chr(10).join(data_lines)}
{frost_context}

Flag anything notable:
- Frost risk (temp_min < {FROST_RISK_THRESHOLD_F}°F)
- Heavy rain (precip > {HEAVY_RAIN_THRESHOLD_IN} in)
- Heat stress (temp_max > {HEAT_STRESS_THRESHOLD_F}°F)
- High wind (wind_max > {HIGH_WIND_THRESHOLD_MPH} mph)

If conditions look good for gardening, say so. Write plain text only — no markdown, no bullet points."""

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_SUMMARY,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = response.choices[0].message.content.strip()

    conn = duckdb.connect(weather_db_path)
    conn.execute(CREATE_SUMMARY_TABLE)
    conn.execute(
        """INSERT INTO weather_forecast_summary (zipcode, summary_text, fetched_at)
           VALUES (?, ?, ?)
           ON CONFLICT (zipcode) DO UPDATE SET
               summary_text = excluded.summary_text,
               fetched_at   = excluded.fetched_at""",
        [zipcode, summary_text, datetime.now(timezone.utc)],
    )
    conn.commit()
    conn.close()

    return {"summary": summary_text}


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_get(lst: list, i: int):
    """Return lst[i] if in bounds, else None."""
    return lst[i] if i < len(lst) else None


def _fmt_wind(degrees) -> str | None:
    """Format wind direction degrees as '270°', or None."""
    if degrees is None:
        return None
    try:
        return f"{int(degrees)}°"
    except (TypeError, ValueError):
        return None
