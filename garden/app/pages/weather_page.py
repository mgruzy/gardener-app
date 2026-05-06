"""Weather page — historical charts, frost dates, sow calendar, and 7-day window."""

import json
import urllib.request
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

from app.db.connections import DatabaseConnections


# ── Parameters ────────────────────────────────────────────────────────────────

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

COLOR_HIGH            = "#e6194b"
COLOR_LOW             = "#4363d8"
COLOR_PRECIP          = "#42d4f4"
COLOR_PRECIP_FORECAST = "#b2ebf2"

DAYS_BACK  = 4
DAYS_AHEAD = 3

FORECAST_SUMMARY_MODEL = "gpt-4o-mini"

PHZM_URL     = "https://phzmapi.org/{zipcode}.json"
ARCHIVE_URL  = (
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


# ── DB loaders ────────────────────────────────────────────────────────────────

def _load_zipcode(db: DatabaseConnections) -> str | None:
    row = db.garden.execute("SELECT zipcode FROM garden LIMIT 1").fetchone()
    return row[0] if row else None


def _load_metadata(db: DatabaseConnections, zipcode: str) -> dict | None:
    row = db.weather.execute(
        "SELECT first_record_date, last_updated_date FROM weather_metadata WHERE zipcode = ?",
        [zipcode],
    ).fetchone()
    if not row:
        return None
    return dict(zip(["first_record_date", "last_updated_date"], row))


def _load_frost(db: DatabaseConnections, zipcode: str) -> dict | None:
    row = db.weather.execute(
        """SELECT avg_last_spring_frost, avg_first_fall_frost, frost_free_days, years_analyzed
           FROM frost_dates WHERE zipcode = ?""",
        [zipcode],
    ).fetchone()
    if not row:
        return None
    return dict(zip(["spring_frost", "fall_frost", "frost_free_days", "years_analyzed"], row))



def _load_climate_patterns(db: DatabaseConnections, zipcode: str) -> dict:
    """Long-run climate pattern percentages from full historical record."""
    def pct(sql: str) -> float | None:
        row = db.weather.execute(sql, [zipcode]).fetchone()
        return row[0] if row else None

    return {
        "high_wind_pct": pct(
            "SELECT ROUND(100.0*SUM(CASE WHEN wind_max_mph>15 THEN 1 ELSE 0 END)/COUNT(*),1)"
            " FROM weather_records WHERE zipcode=? AND wind_max_mph IS NOT NULL"
        ),
        "precip_days_pct": pct(
            "SELECT ROUND(100.0*SUM(CASE WHEN precipitation_in>0.1 THEN 1 ELSE 0 END)/COUNT(*),1)"
            " FROM weather_records WHERE zipcode=? AND precipitation_in IS NOT NULL"
        ),
    }


def _load_monthly_temps(db: DatabaseConnections, zipcode: str) -> pd.DataFrame:
    rows = db.weather.execute(
        """SELECT MONTH(date) AS m,
                  ROUND(AVG(temp_min_f), 1), ROUND(AVG(temp_max_f), 1),
                  ROUND(MIN(temp_min_f), 1), ROUND(MAX(temp_max_f), 1)
           FROM weather_records
           WHERE zipcode = ? AND temp_min_f IS NOT NULL AND temp_max_f IS NOT NULL
           GROUP BY m ORDER BY m""",
        [zipcode],
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["m", "Avg Low", "Avg High", "Record Low", "Record High"])
    df["Month"] = df["m"].map(MONTH_LABELS)
    return df.set_index("Month").drop(columns=["m"])


def _load_monthly_precip(db: DatabaseConnections, zipcode: str) -> pd.DataFrame:
    rows = db.weather.execute(
        """SELECT MONTH(date) AS m,
                  ROUND(SUM(precipitation_in) / COUNT(DISTINCT YEAR(date)), 2)
           FROM weather_records
           WHERE zipcode = ? AND precipitation_in IS NOT NULL
           GROUP BY m ORDER BY m""",
        [zipcode],
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["m", "Avg Precipitation (in)"])
    df["Month"] = df["m"].map(MONTH_LABELS)
    return df.set_index("Month").drop(columns=["m"])


def _load_annual_temps(db: DatabaseConnections, zipcode: str) -> pd.DataFrame:
    rows = db.weather.execute(
        """SELECT YEAR(date) AS yr,
                  ROUND(AVG(temp_min_f), 1), ROUND(AVG(temp_max_f), 1)
           FROM weather_records
           WHERE zipcode = ? AND temp_min_f IS NOT NULL AND temp_max_f IS NOT NULL
           GROUP BY yr ORDER BY yr""",
        [zipcode],
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Year", "Avg Low", "Avg High"])
    return df.set_index("Year")


def _load_historical_for_days(db: DatabaseConnections, zipcode: str, days: list[date]) -> pd.DataFrame:
    """Historical averages (across all years) for each calendar day in the window.

    Groups by month+day so e.g. every May 4 across 20 years is aggregated together.
    """
    mmdd_keys = [d.month * 100 + d.day for d in days]
    placeholders = ", ".join("?" * len(mmdd_keys))
    rows = db.weather.execute(
        f"""SELECT MONTH(date) AS m, DAY(date) AS d,
                  ROUND(AVG(temp_min_f), 1)    AS avg_low,
                  ROUND(AVG(temp_max_f), 1)    AS avg_high,
                  ROUND(MIN(temp_min_f), 1)    AS rec_low,
                  ROUND(MAX(temp_max_f), 1)    AS rec_high,
                  ROUND(AVG(precipitation_in), 2) AS avg_precip
           FROM weather_records
           WHERE zipcode = ?
             AND temp_max_f IS NOT NULL
             AND (MONTH(date) * 100 + DAY(date)) IN ({placeholders})
           GROUP BY m, d
           ORDER BY m, d""",
        [zipcode] + mmdd_keys,
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["m", "d", "Avg Low", "Avg High", "Record Low", "Record High", "Avg Precip"])
    # Map back to the window dates using month+day key
    key_to_date = {dt.month * 100 + dt.day: dt for dt in days}
    df["date"] = (df["m"] * 100 + df["d"]).map(key_to_date)
    df["label"] = df["date"].apply(lambda dt: dt.strftime("%a %-m/%-d") if dt else "")
    return df.dropna(subset=["date"]).sort_values("date")


def _load_forecast_window(db: DatabaseConnections, zipcode: str) -> pd.DataFrame:
    """Load actual + forecast records from weather_forecast for the current 7-day window."""
    today = date.today()
    start = today - timedelta(days=DAYS_BACK)
    end   = today + timedelta(days=DAYS_AHEAD)
    try:
        rows = db.weather.execute(
            """SELECT date, record_type, temp_min_f, temp_max_f, precipitation_in, wind_max_mph
               FROM weather_forecast
               WHERE zipcode = ? AND date BETWEEN ? AND ?
               ORDER BY date""",
            [zipcode, str(start), str(end)],
        ).fetchall()
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=["date", "record_type", "temp_min_f", "temp_max_f", "precipitation_in", "wind_max_mph"],
    )


def _load_forecast_summary(
    db: DatabaseConnections, zipcode: str
) -> tuple[str | None, datetime | None]:
    """Load the AI weather briefing text and its fetch timestamp."""
    try:
        row = db.weather.execute(
            "SELECT summary_text, fetched_at FROM weather_forecast_summary WHERE zipcode = ?",
            [zipcode],
        ).fetchone()
        if row:
            return row[0], row[1]
    except Exception:
        pass
    return None, None


def _load_sow_calendar(db: DatabaseConnections, zipcode: str) -> pd.DataFrame:
    rows = db.weather.execute(
        """SELECT plant_name, indoor_start_date, outdoor_sow_date,
                  estimated_outdoor_sow_date, recommended_date,
                  recommended_source, notes
           FROM sow_calendar WHERE zipcode = ?
           ORDER BY COALESCE(recommended_date, outdoor_sow_date), plant_name""",
        [zipcode],
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "Plant", "Indoor Start", "Outdoor Sow",
        "Estimated Sow", "Recommended Date", "Rec. Source", "Notes",
    ])


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_frost_summary(frost: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Spring Frost", str(frost["spring_frost"]) if frost["spring_frost"] else "n/a")
    c2.metric("First Fall Frost",  str(frost["fall_frost"])   if frost["fall_frost"]   else "n/a")
    c3.metric("Frost-Free Days",   frost["frost_free_days"])
    c4.metric("Years Analyzed",    frost["years_analyzed"])



def _render_climate_patterns(db: DatabaseConnections, zipcode: str) -> None:
    """Long-run climate pattern stats from full historical record."""
    p = _load_climate_patterns(db, zipcode)

    st.subheader("Climate Patterns")
    c1, c2 = st.columns(2)

    def _fmt(val: float | None, suffix: str = "%") -> str:
        return f"{val}{suffix}" if val is not None else "n/a"

    c1.metric("💨 Days with High Wind",     _fmt(p["high_wind_pct"]),
              help="% of days with wind > 15 mph")
    c2.metric("🌧️ Days with Precipitation", _fmt(p["precip_days_pct"]),
              help="% of days with > 0.1 in precipitation")


def _render_zoomed_charts(db: DatabaseConnections, zipcode: str) -> None:
    """7-day window: actual data + forecast overlaid on historical avg band."""
    today = date.today()
    window_days = [today - timedelta(days=DAYS_BACK) + timedelta(days=i)
                   for i in range(DAYS_BACK + 1 + DAYS_AHEAD)]

    hist_df = _load_historical_for_days(db, zipcode, window_days)
    fc_df   = _load_forecast_window(db, zipcode)

    if hist_df.empty and fc_df.empty:
        st.caption("No data — run `build/weather_build/main.py` and `build/weather_forecast/main.py`.")
        return

    # Ordered label list for consistent x-axis
    labels      = [d.strftime("%a %-m/%-d") for d in window_days]
    today_label = today.strftime("%a %-m/%-d")

    # Split forecast frame by record type
    if not fc_df.empty:
        fc_df["label"] = pd.to_datetime(fc_df["date"]).dt.strftime("%a %-m/%-d")
        actual_df   = fc_df[fc_df["record_type"] == "actual"].copy()
        forecast_df = fc_df[fc_df["record_type"] == "forecast"].copy()
        # Bridge: extend actual by the first forecast point so the lines connect
        if not actual_df.empty and not forecast_df.empty:
            actual_df = pd.concat([actual_df, forecast_df.iloc[[0]]], ignore_index=True)
    else:
        actual_df   = pd.DataFrame()
        forecast_df = pd.DataFrame()

    has_fc = not fc_df.empty

    def _today_line(fig: go.Figure) -> go.Figure:
        if today_label in labels:
            idx = labels.index(today_label)
            fig.add_vline(x=idx, line=dict(color="white", width=1, dash="dash"))
            fig.add_annotation(x=idx, y=1, yref="paper", text="today",
                               showarrow=False, font=dict(size=11),
                               xanchor="left", yanchor="top")
        return fig

    tab_t, tab_p = st.tabs(["🌡️ Temp", "🌧️ Precip"])

    with tab_t:
        fig = go.Figure()

        if not hist_df.empty:
            hl = hist_df["label"].tolist()
            # All-time record band (very faint)
            fig.add_trace(go.Scatter(
                x=hl + hl[::-1],
                y=hist_df["Record High"].tolist() + hist_df["Record Low"].tolist()[::-1],
                fill="toself", fillcolor="rgba(100,160,100,0.07)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="All-time range", showlegend=True,
            ))
            # Avg band
            fig.add_trace(go.Scatter(
                x=hl + hl[::-1],
                y=hist_df["Avg High"].tolist() + hist_df["Avg Low"].tolist()[::-1],
                fill="toself", fillcolor="rgba(100,160,100,0.20)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="Avg range", showlegend=True,
            ))
            # Avg lines — faded if real data is present
            avg_opacity = 0.40 if has_fc else 1.0
            avg_width   = 1.5  if has_fc else 2.0
            fig.add_trace(go.Scatter(
                x=hl, y=hist_df["Avg High"], name="Avg High",
                line=dict(color=COLOR_HIGH, width=avg_width), opacity=avg_opacity,
            ))
            fig.add_trace(go.Scatter(
                x=hl, y=hist_df["Avg Low"], name="Avg Low",
                line=dict(color=COLOR_LOW, width=avg_width), opacity=avg_opacity,
            ))

        # Actual measured data — bold solid with filled markers
        if not actual_df.empty:
            fig.add_trace(go.Scatter(
                x=actual_df["label"].tolist(), y=actual_df["temp_max_f"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, color=COLOR_HIGH),
                line=dict(color=COLOR_HIGH, width=2.5),
                name="Actual High",
            ))
            fig.add_trace(go.Scatter(
                x=actual_df["label"].tolist(), y=actual_df["temp_min_f"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, color=COLOR_LOW),
                line=dict(color=COLOR_LOW, width=2.5),
                name="Actual Low",
            ))

        # Forecast — dashed with open markers
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df["label"].tolist(), y=forecast_df["temp_max_f"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, symbol="circle-open", color=COLOR_HIGH),
                line=dict(color=COLOR_HIGH, width=2, dash="dash"),
                name="Forecast High",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["label"].tolist(), y=forecast_df["temp_min_f"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, symbol="circle-open", color=COLOR_LOW),
                line=dict(color=COLOR_LOW, width=2, dash="dash"),
                name="Forecast Low",
            ))

        _today_line(fig)
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=labels),
            yaxis_title="°F",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0), height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
        if has_fc:
            st.caption("Solid = actual measured · Dashed = forecast · Shaded = historical avg range")
        else:
            st.caption("Historical avg high/low + all-time record range for these calendar days.")

    with tab_p:
        fig = go.Figure()

        if not hist_df.empty:
            hl = hist_df["label"].tolist()
            avg_opacity = 0.40 if has_fc else 1.0
            avg_width   = 1.5  if has_fc else 2.0
            fig.add_trace(go.Scatter(
                x=hl, y=hist_df["Avg Precip"].tolist(),
                mode="lines+markers",
                marker=dict(size=5, color=COLOR_PRECIP),
                line=dict(color=COLOR_PRECIP, width=avg_width, dash="dot"),
                name="Avg Precip",
                opacity=avg_opacity,
            ))

        if not actual_df.empty:
            fig.add_trace(go.Scatter(
                x=actual_df["label"].tolist(), y=actual_df["precipitation_in"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, color=COLOR_PRECIP),
                line=dict(color=COLOR_PRECIP, width=2.5),
                name="Actual",
            ))

        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df["label"].tolist(), y=forecast_df["precipitation_in"].tolist(),
                mode="lines+markers",
                marker=dict(size=7, symbol="circle-open", color=COLOR_PRECIP),
                line=dict(color=COLOR_PRECIP, width=2, dash="dash"),
                name="Forecast",
            ))

        _today_line(fig)
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=labels),
            yaxis_title="inches",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0), height=220,
        )
        st.plotly_chart(fig, use_container_width=True)
        if has_fc:
            st.caption("Solid = actual measured · Dashed = forecast · Dotted = historical avg")
        else:
            st.caption("Historical avg precipitation for these calendar days across all recorded years.")


# ── Forecast fetch ────────────────────────────────────────────────────────────

def _geocode(zipcode: str) -> tuple[float | None, float | None]:
    """Fetch lat/lon for a zipcode from the PHZM API."""
    try:
        with urllib.request.urlopen(PHZM_URL.format(zipcode=zipcode), timeout=10) as resp:
            data = json.loads(resp.read())
        coords = data.get("coordinates", {})
        return coords.get("lat"), coords.get("lon")
    except Exception:
        return None, None


def _fetch_forecast_inline(db: DatabaseConnections, zipcode: str) -> tuple[bool, str]:
    """
    Fetch last 4 days of actual weather and next 3-day forecast from Open-Meteo.

    Reads lat/lon from garden.duckdb; falls back to PHZM geocoding if missing.
    Writes records to weather_forecast, generates a 1-2 sentence summary via OpenAI,
    and stores it in weather_forecast_summary. Returns (success, message).
    """
    row = db.garden.execute("SELECT lat, lon FROM garden LIMIT 1").fetchone()
    lat, lon = (row[0], row[1]) if row else (None, None)
    if lat is None or lon is None:
        lat, lon = _geocode(zipcode)
    if lat is None or lon is None:
        return False, f"Could not resolve coordinates for {zipcode}."

    today = date.today()
    start_actual = today - timedelta(days=DAYS_BACK)
    end_actual = today - timedelta(days=1)

    try:
        with urllib.request.urlopen(
            ARCHIVE_URL.format(lat=lat, lon=lon, start=start_actual, end=end_actual),
            timeout=30,
        ) as resp:
            archive = json.loads(resp.read())
    except Exception as e:
        return False, f"Archive fetch failed: {e}"

    try:
        with urllib.request.urlopen(
            FORECAST_URL.format(lat=lat, lon=lon, days=DAYS_AHEAD + 1),
            timeout=30,
        ) as resp:
            forecast = json.loads(resp.read())
    except Exception as e:
        return False, f"Forecast fetch failed: {e}"

    records: list[dict] = []
    for daily, rtype in [(archive.get("daily", {}), "actual"), (forecast.get("daily", {}), "forecast")]:
        for i, d_str in enumerate(daily.get("time", [])):
            wdir = daily.get("wind_direction_10m_dominant", [])
            records.append({
                "date": d_str,
                "record_type": rtype,
                "temp_min_f": daily.get("temperature_2m_min", [None] * (i + 1))[i],
                "temp_max_f": daily.get("temperature_2m_max", [None] * (i + 1))[i],
                "precipitation_in": daily.get("precipitation_sum", [None] * (i + 1))[i],
                "wind_max_mph": daily.get("wind_speed_10m_max", [None] * (i + 1))[i],
                "wind_direction": f"{int(wdir[i])}°" if i < len(wdir) and wdir[i] is not None else None,
            })

    fetched_at = datetime.now()
    db.weather.execute("DELETE FROM weather_forecast WHERE zipcode = ?", [zipcode])
    for r in records:
        db.weather.execute(
            """INSERT INTO weather_forecast
                   (zipcode, date, record_type, temp_min_f, temp_max_f,
                    precipitation_in, wind_max_mph, wind_direction, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [zipcode, r["date"], r["record_type"],
             r["temp_min_f"], r["temp_max_f"],
             r["precipitation_in"], r["wind_max_mph"],
             r["wind_direction"], fetched_at],
        )
    db.weather.commit()

    # 1-2 sentence summary with above/below average context
    frost = _load_frost(db, zipcode)
    window_days = [today - timedelta(days=DAYS_BACK) + timedelta(days=i)
                   for i in range(DAYS_BACK + 1 + DAYS_AHEAD)]
    hist_df = _load_historical_for_days(db, zipcode, window_days)

    data_lines = [
        f"{r['date']} ({r['record_type']}): high={r['temp_max_f']}°F  "
        f"low={r['temp_min_f']}°F  precip={r['precipitation_in']}in  wind={r['wind_max_mph']}mph"
        for r in records
    ]
    hist_lines: list[str] = []
    if not hist_df.empty:
        hist_lines = [
            f"{row['label']}: avg_high={row['Avg High']}°F  avg_low={row['Avg Low']}°F  avg_precip={row['Avg Precip']}in"
            for _, row in hist_df.iterrows()
        ]
    frost_ctx = f"\nAvg last spring frost: {frost['spring_frost']}" if frost else ""

    hist_block = ("\n\nHistorical averages for these same calendar days (20-yr):\n"
                  + "\n".join(hist_lines)) if hist_lines else ""

    prompt = (
        "Garden weather assistant. In 1-2 plain sentences, tell a gardener what to know "
        "about this week. Mention whether temperatures and precipitation are above or below "
        "the 20-year historical average for these dates. Flag frost (<36°F), heavy rain "
        "(>0.5in), heat (>85°F), or high wind (>15mph). If all clear, say so. No markdown.\n\n"
        "Forecast/actual data:\n"
        + "\n".join(data_lines)
        + hist_block
        + frost_ctx
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=FORECAST_SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = resp.choices[0].message.content.strip()

    db.weather.execute(
        """INSERT INTO weather_forecast_summary (zipcode, summary_text, fetched_at)
           VALUES (?, ?, ?)
           ON CONFLICT (zipcode) DO UPDATE SET
               summary_text = excluded.summary_text,
               fetched_at   = excluded.fetched_at""",
        [zipcode, summary, fetched_at],
    )
    db.weather.commit()

    actual_n   = sum(1 for r in records if r["record_type"] == "actual")
    forecast_n = sum(1 for r in records if r["record_type"] == "forecast")
    return True, f"Loaded {actual_n} actual + {forecast_n} forecast days."


def _render_weather_bot(db: DatabaseConnections, zipcode: str) -> None:
    """Stale-data notice if no forecast has been fetched yet."""
    pass


def _render_charts(db: DatabaseConnections, zipcode: str) -> None:
    tab_temp, tab_precip, tab_annual = st.tabs(
        ["🌡️ Monthly Temps", "🌧️ Precipitation", "📈 Annual Trend"]
    )

    with tab_temp:
        df = _load_monthly_temps(db, zipcode)
        if df.empty:
            st.info("No temperature records.")
        else:
            months = df.index.tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months + months[::-1],
                y=df["Avg High"].tolist() + df["Avg Low"].tolist()[::-1],
                fill="toself", fillcolor="rgba(100,160,100,0.15)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(x=months, y=df["Avg High"], name="Avg High",
                                     line=dict(color=COLOR_HIGH, width=2)))
            fig.add_trace(go.Scatter(x=months, y=df["Avg Low"],  name="Avg Low",
                                     line=dict(color=COLOR_LOW,  width=2)))
            fig.add_trace(go.Scatter(x=months, y=df["Record High"], name="Record High",
                                     line=dict(color=COLOR_HIGH, width=1, dash="dot"), opacity=0.5))
            fig.add_trace(go.Scatter(x=months, y=df["Record Low"],  name="Record Low",
                                     line=dict(color=COLOR_LOW,  width=1, dash="dot"), opacity=0.5))
            fig.update_layout(yaxis_title="°F",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              margin=dict(l=0, r=0, t=40, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Solid = 20-yr monthly average · Dotted = all-time record")

    with tab_precip:
        df = _load_monthly_precip(db, zipcode)
        if df.empty:
            st.info("No precipitation records.")
        else:
            fig = go.Figure(go.Bar(
                x=df.index.tolist(), y=df["Avg Precipitation (in)"],
                marker_color=COLOR_PRECIP,
            ))
            fig.update_layout(yaxis_title="inches", margin=dict(l=0, r=0, t=30, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Average monthly precipitation across all recorded years.")

    with tab_annual:
        df = _load_annual_temps(db, zipcode)
        if df.empty:
            st.info("No annual records.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Avg High"], name="Avg High",
                                     line=dict(color=COLOR_HIGH, width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df["Avg Low"],  name="Avg Low",
                                     line=dict(color=COLOR_LOW,  width=2)))
            fig.update_layout(yaxis_title="°F",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Yearly average of daily highs and lows.")


def _render_sow_calendar(db: DatabaseConnections, zipcode: str) -> None:
    df = _load_sow_calendar(db, zipcode)
    if df.empty:
        st.info("No sow calendar — run `build/weather_build/main.py` first.")
        return

    st.subheader("🌱 Sow Calendar")
    today = date.today()

    def _status(row: pd.Series) -> str:
        d = row["Recommended Date"] or row["Outdoor Sow"]
        if d is None:
            return ""
        try:
            d = date.fromisoformat(str(d))
        except ValueError:
            return ""
        delta = (d - today).days
        if delta < -7:
            return "⏰ Overdue"
        if delta <= 14:
            return "🔜 This week/next"
        if delta <= 45:
            return "📅 Coming up"
        return "🗓️ Future"

    df["Timing"] = df.apply(_status, axis=1)

    search = st.text_input("Filter plants", placeholder="e.g. tomato", key="sow_search")
    if search:
        df = df[df["Plant"].str.contains(search, case=False, na=False)]

    st.dataframe(
        df[["Plant", "Timing", "Indoor Start", "Outdoor Sow",
            "Estimated Sow", "Recommended Date", "Rec. Source", "Notes"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Indoor Start":     st.column_config.DateColumn("Indoor Start"),
            "Outdoor Sow":      st.column_config.DateColumn("Outdoor Sow"),
            "Estimated Sow":    st.column_config.DateColumn("Estimated Sow"),
            "Recommended Date": st.column_config.DateColumn("Recommended Date"),
        },
    )
    st.caption(f"{len(df)} plants · Timing relative to today ({today})")


def _render_weather_refresh(db: DatabaseConnections, zipcode: str) -> None:
    """Weather refresh button expander."""
    with st.expander("🔧 Maintenance", expanded=False):
        if st.button("🌤️ Refresh Weather", use_container_width=True):
            with st.spinner("Fetching weather…"):
                ok, msg = _fetch_forecast_inline(db, zipcode)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def render(db: DatabaseConnections) -> None:
    zipcode = _load_zipcode(db)
    if not zipcode:
        st.info("Set up your garden on the Map tab first.")
        return

    meta  = _load_metadata(db, zipcode)
    frost = _load_frost(db, zipcode)

    st.subheader(f"🌤️ Weather — {zipcode}")

    if not frost:
        st.info("No frost data — run `build/weather_build/main.py` first.")
        return

    _render_weather_refresh(db, zipcode)
    st.divider()

    st.subheader("📅 Recent & Upcoming")
    _render_weather_bot(db, zipcode)

    summary, fc_fetched_at = _load_forecast_summary(db, zipcode)
    if summary:
        st.info(f"**Briefing:** {summary}")
    else:
        st.caption("No forecast yet — use 🔧 Maintenance → Refresh Weather below.")

    _render_zoomed_charts(db, zipcode)

    if fc_fetched_at:
        try:
            ts = pd.Timestamp(fc_fetched_at).strftime("%b %-d at %-I:%M %p")
        except Exception:
            ts = str(fc_fetched_at)
        st.caption(f"Forecast last fetched: {ts}  ·  Use 🔧 Maintenance to refresh.")

    st.divider()

    _render_climate_patterns(db, zipcode)
    st.divider()

    with st.expander("📊 Historical Charts", expanded=False):
        _render_charts(db, zipcode)

    st.divider()
    _render_frost_summary(frost)
    if meta:
        st.caption(f"Records: {meta['first_record_date']} → {meta['last_updated_date']}")
    st.divider()
    _render_sow_calendar(db, zipcode)
