"""All @function_tool definitions for the weather_build agents."""

import json
import urllib.request
from datetime import date, timedelta
from uuid import uuid4

import duckdb
import final_judge
import sow_estimator
from agents import function_tool
from openai import OpenAI

# --- Parameters ---
HISTORY_YEARS = 20
FROST_THRESHOLD_F = 32.0
YEARS_TO_ANALYZE = 10
SAFE_PLANTING_OFFSET_WEEKS = 2


# --- Weather History Tools ---

@function_tool
def geocode_zipcode(zipcode: str) -> dict:
    """Get latitude, longitude, and hardiness zone for a US zipcode using the PHZM API."""
    url = f"https://phzmapi.org/{zipcode}.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        coords = data.get("coordinates", {})
        return {
            "zipcode": zipcode,
            "lat": coords.get("lat"),
            "lon": coords.get("lon"),
            "zone": data.get("zone"),
        }
    except Exception as e:
        return {"error": str(e)}


@function_tool
def fetch_and_write_weather(zipcode: str, lat: float, lon: float, db_path: str) -> dict:
    """
    Fetch 20 years of daily weather from the Open-Meteo archive API and write to weather.duckdb.

    Fetches temperature, precipitation, and wind data in imperial units.
    Skips dates already present in the database. Returns a write summary.
    """
    end_date = date.today() - timedelta(days=1)
    start_date = date(end_date.year - HISTORY_YEARS, 1, 1)

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"wind_speed_10m_max,wind_direction_10m_dominant"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
        f"&timezone=auto"
    )

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"error": f"Open-Meteo fetch failed: {e}"}

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    wind_max = daily.get("wind_speed_10m_max", [])
    wind_dir = daily.get("wind_direction_10m_dominant", [])

    conn = duckdb.connect(db_path)
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

    # TODO -> make this part of the prompt -- we can do a checker first and make a set and
    # send those into only -> for now this is a simple code loop
    written = 0
    skipped = 0
    for i, d in enumerate(dates):
        existing = conn.execute(
            "SELECT 1 FROM weather_records WHERE zipcode = ? AND date = ?",
            [zipcode, d],
        ).fetchone()
        if existing:
            skipped += 1
            continue

        conn.execute(
            """
            INSERT INTO weather_records
                (zipcode, date, temp_min_f, temp_max_f, precipitation_in,
                 wind_max_mph, wind_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                zipcode,
                d,
                temp_min[i] if i < len(temp_min) else None,
                temp_max[i] if i < len(temp_max) else None,
                precip[i] if i < len(precip) else None,
                wind_max[i] if i < len(wind_max) else None,
                f"{int(wind_dir[i])}°" if i < len(wind_dir) and wind_dir[i] is not None else None,
            ],
        )
        written += 1

    conn.execute(
        """
        INSERT INTO weather_metadata (zipcode, first_record_date, last_updated_date)
        VALUES (?, ?, ?)
        ON CONFLICT (zipcode) DO UPDATE SET last_updated_date = excluded.last_updated_date
        """,
        [zipcode, str(start_date), str(end_date)],
    )
    conn.close()

    return {
        "zipcode": zipcode,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "records_written": written,
        "records_skipped": skipped,
    }


# --- Frost Date Tools ---

@function_tool
def calculate_and_write_frost_dates(zipcode: str, weather_db_path: str) -> dict:
    """
    Analyze weather history to find average last spring frost and first fall frost.

    Uses the last 10 years of records. For frost-free climates (e.g. zone 9b+),
    returns frost_free_days=365 and null frost dates. Writes results to the
    frost_dates table in weather.duckdb.
    """
    conn = duckdb.connect(weather_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS frost_dates (
            zipcode                 VARCHAR PRIMARY KEY,
            avg_last_spring_frost   DATE,
            avg_first_fall_frost    DATE,
            frost_free_days         INTEGER,
            years_analyzed          INTEGER
        )
    """)

    count = conn.execute(
        "SELECT COUNT(*) FROM weather_records WHERE zipcode = ?", [zipcode]
    ).fetchone()[0]

    if count == 0:
        conn.close()
        return {"error": f"No weather records for zipcode {zipcode} — run weather_history first."}

    spring_frosts = conn.execute(
        """
        SELECT YEAR(date) AS yr, MAX(date) AS last_frost
        FROM weather_records
        WHERE zipcode = ?
          AND MONTH(date) BETWEEN 1 AND 6
          AND temp_min_f <= 32.0
          AND YEAR(date) >= YEAR(CURRENT_DATE) - ?
        GROUP BY YEAR(date)
        ORDER BY yr
        """,
        [zipcode, YEARS_TO_ANALYZE],
    ).fetchall()

    fall_frosts = conn.execute(
        """
        SELECT YEAR(date) AS yr, MIN(date) AS first_frost
        FROM weather_records
        WHERE zipcode = ?
          AND MONTH(date) BETWEEN 7 AND 12
          AND temp_min_f <= 32.0
          AND YEAR(date) >= YEAR(CURRENT_DATE) - ?
        GROUP BY YEAR(date)
        ORDER BY yr
        """,
        [zipcode, YEARS_TO_ANALYZE],
    ).fetchall()

    current_year = date.today().year

    if not spring_frosts:
        avg_last_spring = None
        avg_first_fall = None
        frost_free_days = 365
        years_analyzed = 0
    else:
        spring_doys = [d.timetuple().tm_yday for _, d in spring_frosts if d]
        fall_doys = [d.timetuple().tm_yday for _, d in fall_frosts if d]

        avg_spring_doy = int(sum(spring_doys) / len(spring_doys)) if spring_doys else None
        avg_fall_doy = int(sum(fall_doys) / len(fall_doys)) if fall_doys else None

        avg_last_spring = (
            str(date(current_year, 1, 1) + timedelta(days=avg_spring_doy - 1))
            if avg_spring_doy else None
        )
        avg_first_fall = (
            str(date(current_year, 1, 1) + timedelta(days=avg_fall_doy - 1))
            if avg_fall_doy else None
        )
        frost_free_days = (
            (avg_fall_doy - avg_spring_doy) if avg_spring_doy and avg_fall_doy else 365
        )
        years_analyzed = len(spring_frosts)

    conn.execute(
        """
        INSERT INTO frost_dates
            (zipcode, avg_last_spring_frost, avg_first_fall_frost, frost_free_days, years_analyzed)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (zipcode) DO UPDATE SET
            avg_last_spring_frost = excluded.avg_last_spring_frost,
            avg_first_fall_frost  = excluded.avg_first_fall_frost,
            frost_free_days       = excluded.frost_free_days,
            years_analyzed        = excluded.years_analyzed
        """,
        [zipcode, avg_last_spring, avg_first_fall, frost_free_days, years_analyzed],
    )
    conn.close()

    return {
        "zipcode": zipcode,
        "avg_last_spring_frost": avg_last_spring,
        "avg_first_fall_frost": avg_first_fall,
        "frost_free_days": frost_free_days,
        "years_analyzed": years_analyzed,
        "note": "Frost-free climate detected — no frost days found." if not spring_frosts else None,
    }


@function_tool
def research_sow_dates(zipcode: str, zone: str, city: str, plant_db_path: str) -> dict:
    """
    Research regionally specific outdoor sow dates for every plant in plant.duckdb.

    Makes one LLM call with all plant names and returns a dict mapping plant_name to
    {outdoor_sow_date, confidence, note}. confidence='high' means a specific regional
    date was found and should be used; confidence='low' means the date is too general
    and the frost-calculation fallback should be used instead.
    """
    p_conn = duckdb.connect(plant_db_path, read_only=True)
    plant_names = [
        row[0] for row in p_conn.execute("SELECT name FROM plant_types ORDER BY name").fetchall()
    ]
    p_conn.close()

    current_year = date.today().year
    client = OpenAI()

    prompt = f"""You are an expert horticulturalist. For each plant below, provide the recommended outdoor planting date for:
- Zipcode: {zipcode}
- City: {city}
- Hardiness Zone: {zone}
- Year: {current_year}

Return a JSON object where each key is the exact plant name and each value has:
- "outdoor_sow_date": specific date as "{current_year}-MM-DD", or null if truly too variable to pin down
- "confidence": "high" if you have a specific regional recommendation for this zone/city, "low" if too general
- "note": one sentence explaining the date (e.g. "soil needs 60F, typically late May in Seattle")

Be specific to the zone and city — do not just say "after last frost."
Plants:
{chr(10).join(f"- {p}" for p in plant_names)}

Return only valid JSON, no markdown."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def estimate_sow_dates(zipcode: str, weather_db_path: str, plant_db_path: str) -> str:
    """
    Run a two-agent pipeline to estimate outdoor sow dates from plant database ranges.

    A DB Reader Agent pulls plant sow date ranges, frost dates, and monthly avg temps.
    A Date Researcher Agent interprets each range for the specific location and computes
    a specific estimated_outdoor_sow_date. Plants with range 'n/a' return null.
    Returns a JSON string mapping plant_name to estimated date (YYYY-MM-DD or null).
    """
    estimates = await sow_estimator.run_async(zipcode, weather_db_path, plant_db_path)
    return json.dumps(estimates)


@function_tool
def build_sow_calendar(
    zipcode: str,
    weather_db_path: str,
    plant_db_path: str,
    researched_dates_json: str,
    estimated_dates_json: str,
) -> dict:
    """
    Build a per-plant sow calendar combining researched and estimated outdoor sow dates.

    outdoor_sow_date uses researched_dates_json when confidence='high', otherwise
    falls back to last_spring_frost + 2 weeks. estimated_outdoor_sow_date comes from
    the two-agent range estimator (null when plant range is 'n/a').
    indoor_start_date uses last_spring_frost minus indoor_sow_weeks_before_frost.
    Writes to sow_calendar in weather.duckdb, replacing prior entries for this zipcode.
    """
    try:
        researched = json.loads(researched_dates_json)
    except Exception:
        researched = {}

    try:
        estimated = json.loads(estimated_dates_json)
    except Exception:
        estimated = {}

    w_conn = duckdb.connect(weather_db_path)
    p_conn = duckdb.connect(plant_db_path, read_only=True)

    frost_row = w_conn.execute(
        "SELECT avg_last_spring_frost FROM frost_dates WHERE zipcode = ?", [zipcode]
    ).fetchone()

    if not frost_row:
        w_conn.close()
        p_conn.close()
        return {"error": "No frost dates found — run calculate_and_write_frost_dates first."}

    last_spring_frost = frost_row[0]

    plants = p_conn.execute(
        """
        SELECT pt.plant_type_id, pt.name,
               pv.indoor_sow_weeks_before_frost,
               pv.outdoor_sow_date_range
        FROM plant_types pt
        JOIN plant_varieties pv ON pt.plant_type_id = pv.plant_type_id
        """
    ).fetchall()
    p_conn.close()

    w_conn.execute("""
        CREATE TABLE IF NOT EXISTS sow_calendar (
            sow_id                      VARCHAR PRIMARY KEY,
            zipcode                     VARCHAR NOT NULL,
            plant_type_id               VARCHAR NOT NULL,
            plant_name                  VARCHAR NOT NULL,
            indoor_start_date           DATE,
            outdoor_sow_date            DATE,
            estimated_outdoor_sow_date  DATE,
            recommended_date            DATE,
            recommended_source          VARCHAR,
            sow_source                  VARCHAR,
            notes                       VARCHAR
        )
    """)
    # Migrate existing tables missing columns added after initial creation
    w_conn.execute(
        "ALTER TABLE sow_calendar ADD COLUMN IF NOT EXISTS estimated_outdoor_sow_date DATE"
    )
    w_conn.execute(
        "ALTER TABLE sow_calendar ADD COLUMN IF NOT EXISTS sow_source VARCHAR"
    )
    w_conn.execute(
        "ALTER TABLE sow_calendar ADD COLUMN IF NOT EXISTS recommended_date DATE"
    )
    w_conn.execute(
        "ALTER TABLE sow_calendar ADD COLUMN IF NOT EXISTS recommended_source VARCHAR"
    )
    w_conn.execute("DELETE FROM sow_calendar WHERE zipcode = ?", [zipcode])

    written = 0
    researched_count = 0
    fallback_count = 0
    disagreements = []

    for plant_type_id, name, indoor_weeks, outdoor_range in plants:
        indoor_start = None
        outdoor_sow = None
        sow_source = "fallback"

        if last_spring_frost and indoor_weeks:
            indoor_start = last_spring_frost - timedelta(weeks=int(indoor_weeks))

        research = researched.get(name, {})
        if research.get("confidence") == "high" and research.get("outdoor_sow_date"):
            try:
                outdoor_sow = date.fromisoformat(research["outdoor_sow_date"])
                sow_source = "researched"
                researched_count += 1
            except ValueError:
                pass

        if outdoor_sow is None:
            if last_spring_frost:
                outdoor_sow = last_spring_frost + timedelta(weeks=SAFE_PLANTING_OFFSET_WEEKS)
            sow_source = "fallback"
            fallback_count += 1

        estimated_sow_raw = estimated.get(name)
        try:
            estimated_sow = date.fromisoformat(estimated_sow_raw) if estimated_sow_raw else None
        except ValueError:
            estimated_sow = None

        if outdoor_sow and estimated_sow:
            gap_days = abs((outdoor_sow - estimated_sow).days)
            if gap_days > 14:
                disagreements.append({
                    "plant": name,
                    "outdoor_sow_date": str(outdoor_sow),
                    "estimated_outdoor_sow_date": str(estimated_sow),
                    "gap_days": gap_days,
                    "sow_source": sow_source,
                })

        note_parts = []
        if research.get("note"):
            note_parts.append(research["note"])
        if outdoor_range:
            note_parts.append(f"General range: {outdoor_range}")
        notes = " | ".join(note_parts) if note_parts else None

        w_conn.execute(
            """
            INSERT INTO sow_calendar
                (sow_id, zipcode, plant_type_id, plant_name,
                 indoor_start_date, outdoor_sow_date, estimated_outdoor_sow_date,
                 sow_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [str(uuid4()), zipcode, plant_type_id, name,
             indoor_start, outdoor_sow, estimated_sow, sow_source, notes],
        )
        written += 1

    w_conn.close()
    return {
        "status": "written",
        "zipcode": zipcode,
        "plants_in_calendar": written,
        "researched_dates_used": researched_count,
        "fallback_dates_used": fallback_count,
        "disagreements": disagreements,
    }


@function_tool
async def judge_sow_dates(zipcode: str, weather_db_path: str) -> dict:
    """
    Run the Final Judge Agent to resolve disagreements between researched and estimated sow dates.

    Queries sow_calendar for plants where outdoor_sow_date and estimated_outdoor_sow_date
    differ by more than 14 days. Passes each disagreement to final_judge_agent with full
    context (both dates, sow_source, range). Writes recommended_date and recommended_source
    back to sow_calendar. Returns a summary of judgments made.
    """
    conn = duckdb.connect(weather_db_path)

    rows = conn.execute(
        """
        SELECT plant_name,
               outdoor_sow_date,
               estimated_outdoor_sow_date,
               sow_source,
               notes,
               ABS(DATEDIFF('day', outdoor_sow_date, estimated_outdoor_sow_date)) AS gap_days
        FROM sow_calendar
        WHERE zipcode = ?
          AND outdoor_sow_date IS NOT NULL
          AND estimated_outdoor_sow_date IS NOT NULL
          AND ABS(DATEDIFF('day', outdoor_sow_date, estimated_outdoor_sow_date)) > 14
        ORDER BY gap_days DESC
        """,
        [zipcode],
    ).fetchall()

    if not rows:
        conn.close()
        return {"status": "no_disagreements", "judgments_made": 0}

    lines = [f"Plants with disagreements for zipcode {zipcode}:\n"]
    for plant, outdoor, estimated, source, notes, gap in rows:
        lines.append(
            f"- {plant}\n"
            f"  outdoor_sow_date (sow_source={source}): {outdoor}\n"
            f"  estimated_outdoor_sow_date: {estimated}\n"
            f"  gap: {gap} days\n"
            f"  range/notes: {notes or 'n/a'}\n"
        )

    summary = "\n".join(lines)
    judgments = await final_judge.run_async(summary)

    for plant_name, judgment in judgments.items():
        try:
            rec_date = date.fromisoformat(judgment["recommended_date"])
        except (ValueError, KeyError):
            continue
        conn.execute(
            """
            UPDATE sow_calendar
            SET recommended_date = ?, recommended_source = ?
            WHERE zipcode = ? AND plant_name = ?
            """,
            [rec_date, judgment["chosen_source"], zipcode, plant_name],
        )

    conn.close()
    return {
        "status": "judged",
        "disagreements_reviewed": len(rows),
        "judgments_made": len(judgments),
    }
