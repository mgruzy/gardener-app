"""Plants page — zoomed plot view with plant instance placement (single & row sow styles)."""

import asyncio
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import app.utils.canvas_utils as cu
from app.db.connections import DatabaseConnections
import app.internal_objects.garden_types as garden_enum
import build.soil_amendment.amendment_agent as amendment_agent
import app.pages.layout_optimizer_page as layout_optimizer


# ── Parameters ────────────────────────────────────────────────────────────────

DEFAULT_CANVAS_WIDTH = 1000
INSPECT_POINT_RADIUS = 4
INSPECT_HIT_THRESHOLD_PX = 30.0


# ── Schema migrations ─────────────────────────────────────────────────────────

def _ensure_migrations(db: DatabaseConnections) -> None:
    db.garden.execute("ALTER TABLE plant_instances ADD COLUMN IF NOT EXISTS sow_style VARCHAR DEFAULT 'single'")
    db.garden.execute("ALTER TABLE plant_instances ADD COLUMN IF NOT EXISTS row_id VARCHAR")
    db.garden.execute("ALTER TABLE plant_instances ADD COLUMN IF NOT EXISTS variety_id VARCHAR")
    db.garden.execute("ALTER TABLE plant_instances ADD COLUMN IF NOT EXISTS spacing_inches DOUBLE")
    db.garden.execute("""
        CREATE TABLE IF NOT EXISTS harvest_log (
            harvest_id   VARCHAR PRIMARY KEY,
            instance_id  VARCHAR NOT NULL,
            harvest_date DATE NOT NULL,
            quantity     INTEGER,
            weight_lbs   DOUBLE,
            notes        VARCHAR
        )
    """)
    db.garden.execute("ALTER TABLE harvest_log ADD COLUMN IF NOT EXISTS quantity INTEGER")
    db.garden.execute("""
        CREATE TABLE IF NOT EXISTS soil_amendment_advice (
            advice_id       VARCHAR PRIMARY KEY,
            plot_id         VARCHAR NOT NULL,
            generated_at    TIMESTAMP NOT NULL,
            advice_text     VARCHAR NOT NULL
        )
    """)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_garden(db: DatabaseConnections) -> dict | None:
    row = db.garden.execute(
        "SELECT garden_id, name, zipcode, aerial_image_path, scale_px_per_foot FROM garden LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return dict(zip(["garden_id", "name", "zipcode", "aerial_image_path", "scale_px_per_foot"], row))


def _load_plots(db: DatabaseConnections, garden_id: str) -> list[dict]:
    rows = db.garden.execute(
        "SELECT plot_id, name, area_sqft FROM plots WHERE garden_id = ? ORDER BY name",
        [garden_id],
    ).fetchall()
    return [dict(zip(["plot_id", "name", "area_sqft"], r)) for r in rows]


def _load_plot_full(db: DatabaseConnections, plot_id: str) -> dict | None:
    row = db.garden.execute(
        "SELECT plot_id, name, polygon FROM plots WHERE plot_id = ?", [plot_id]
    ).fetchone()
    if not row:
        return None
    return dict(zip(["plot_id", "name", "polygon"], row))


def _load_regions(db: DatabaseConnections, plot_id: str) -> list[dict]:
    rows = db.garden.execute(
        "SELECT region_type, zone_value, polygon FROM plot_regions WHERE plot_id = ?",
        [plot_id],
    ).fetchall()
    return [dict(zip(["region_type", "zone_value", "polygon"], r)) for r in rows]


def _load_instances(db: DatabaseConnections, plot_id: str) -> list[dict]:
    rows = db.garden.execute(
        """SELECT instance_id, plant_type_id, variety_id, sow_style, row_id,
                  location_x, location_y, spacing_inches, status,
                  planned_sow_date, planted_date
           FROM plant_instances
           WHERE plot_id = ?
             AND status != 'archived'
           ORDER BY planted_date NULLS LAST, planned_sow_date NULLS LAST""",
        [plot_id],
    ).fetchall()
    cols = ["instance_id", "plant_type_id", "variety_id", "sow_style", "row_id",
            "location_x", "location_y", "spacing_inches", "status",
            "planned_sow_date", "planted_date"]
    return [dict(zip(cols, r)) for r in rows]


def _load_plant_types(db: DatabaseConnections) -> list[dict]:
    rows = db.plant.execute(
        "SELECT plant_type_id, name, category FROM plant_types ORDER BY category, name"
    ).fetchall()
    return [dict(zip(["plant_type_id", "name", "category"], r)) for r in rows]


def _load_varieties(db: DatabaseConnections, plant_type_id: str) -> list[dict]:
    rows = db.plant.execute(
        """SELECT variety_id, variety_name, spacing_inches, height_inches_estimate, sun_tolerance
           FROM plant_varieties WHERE plant_type_id = ? ORDER BY variety_name""",
        [plant_type_id],
    ).fetchall()
    cols = ["variety_id", "variety_name", "spacing_inches", "height_inches_estimate", "sun_tolerance"]
    return [dict(zip(cols, r)) for r in rows]


def _load_variety_by_id(db: DatabaseConnections, variety_id: str) -> dict | None:
    if not variety_id:
        return None
    row = db.plant.execute(
        """SELECT variety_id, variety_name, spacing_inches, height_inches_estimate,
                  sun_tolerance, days_to_harvest
           FROM plant_varieties WHERE variety_id = ?""",
        [variety_id],
    ).fetchone()
    if not row:
        return None
    return dict(zip(
        ["variety_id", "variety_name", "spacing_inches", "height_inches_estimate",
         "sun_tolerance", "days_to_harvest"],
        row,
    ))


def _load_days_to_harvest(db: DatabaseConnections, plant_type_id: str) -> int | None:
    """Return days_to_harvest for a plant type, averaged across varieties if multiple exist."""
    row = db.plant.execute(
        """SELECT CAST(AVG(days_to_harvest) AS INTEGER)
           FROM plant_varieties
           WHERE plant_type_id = ? AND days_to_harvest IS NOT NULL""",
        [plant_type_id],
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _load_harvest_log(db: DatabaseConnections, instance_id: str) -> list[dict]:
    rows = db.garden.execute(
        """SELECT harvest_id, harvest_date, quantity, weight_lbs, notes
           FROM harvest_log WHERE instance_id = ? ORDER BY harvest_date DESC""",
        [instance_id],
    ).fetchall()
    return [dict(zip(["harvest_id", "harvest_date", "quantity", "weight_lbs", "notes"], r)) for r in rows]


def _load_row_harvest_log(db: DatabaseConnections, row_id: str) -> list[dict]:
    """Load all harvests for every instance in a row, combined and sorted by date."""
    rows = db.garden.execute(
        """SELECT hl.harvest_id, hl.harvest_date, hl.quantity, hl.weight_lbs, hl.notes
           FROM harvest_log hl
           JOIN plant_instances pi ON pi.instance_id = hl.instance_id
           WHERE pi.row_id = ?
           ORDER BY hl.harvest_date DESC""",
        [row_id],
    ).fetchall()
    return [dict(zip(["harvest_id", "harvest_date", "quantity", "weight_lbs", "notes"], r)) for r in rows]


def _sync_harvest_totals(db: DatabaseConnections, instance_id: str) -> None:
    count = db.garden.execute(
        "SELECT COUNT(*) FROM harvest_log WHERE instance_id = ?", [instance_id]
    ).fetchone()[0]
    total_w = db.garden.execute(
        "SELECT COALESCE(SUM(weight_lbs), 0.0) FROM harvest_log WHERE instance_id = ? AND weight_lbs IS NOT NULL",
        [instance_id],
    ).fetchone()[0]
    db.garden.execute(
        "UPDATE plant_instances SET harvest_count = ?, harvest_weight_lbs = ? WHERE instance_id = ?",
        [count, total_w, instance_id],
    )
    db.garden.commit()


def _add_harvest(
    db: DatabaseConnections,
    instance_id: str,
    harvest_date: date,
    quantity: int | None,
    weight_lbs: float | None,
    notes: str | None,
) -> None:
    db.garden.execute(
        """INSERT INTO harvest_log (harvest_id, instance_id, harvest_date, quantity, weight_lbs, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [str(uuid4()), instance_id, harvest_date, quantity, weight_lbs, notes or None],
    )
    db.garden.commit()
    _sync_harvest_totals(db, instance_id)


def _delete_harvest(db: DatabaseConnections, harvest_id: str, instance_id: str) -> None:
    db.garden.execute("DELETE FROM harvest_log WHERE harvest_id = ?", [harvest_id])
    db.garden.commit()
    _sync_harvest_totals(db, instance_id)


def _archive_instance(db: DatabaseConnections, instance_id: str) -> None:
    """Set status to 'archived' — hidden from plot view, retained in DB."""
    db.garden.execute(
        "UPDATE plant_instances SET status = 'archived' WHERE instance_id = ?",
        [instance_id],
    )
    db.garden.commit()


def _archive_row(db: DatabaseConnections, row_id: str) -> None:
    """Archive all instances in a row."""
    db.garden.execute(
        "UPDATE plant_instances SET status = 'archived' WHERE row_id = ?",
        [row_id],
    )
    db.garden.commit()


def _count_archived(db: DatabaseConnections, plot_id: str) -> int:
    row = db.garden.execute(
        "SELECT COUNT(*) FROM plant_instances WHERE plot_id = ? AND status = 'archived'",
        [plot_id],
    ).fetchone()
    return row[0] if row else 0


def _update_instance_status(db: DatabaseConnections, instance_id: str, new_status: str) -> None:
    db.garden.execute(
        "UPDATE plant_instances SET status = ? WHERE instance_id = ?",
        [new_status, instance_id],
    )
    db.garden.commit()


def _update_instance_variety(
    db: DatabaseConnections,
    instance_id: str,
    variety_id: str,
    spacing_inches: float,
) -> None:
    db.garden.execute(
        "UPDATE plant_instances SET variety_id = ?, spacing_inches = ? WHERE instance_id = ?",
        [variety_id, spacing_inches, instance_id],
    )
    db.garden.commit()


def _update_instance_spacing(
    db: DatabaseConnections, instance_id: str, spacing_inches: float
) -> None:
    db.garden.execute(
        "UPDATE plant_instances SET spacing_inches = ? WHERE instance_id = ?",
        [spacing_inches, instance_id],
    )
    db.garden.commit()


def _save_instances(
    db: DatabaseConnections,
    plot_id: str,
    plant_type_id: str,
    variety_id: str,
    sow_style: str,
    status: str,
    planned_sow_date: date | None,
    positions_orig: list[tuple[float, float]],
    spacing_inches: float,
) -> None:
    row_id = str(uuid4()) if sow_style == "row" and len(positions_orig) > 1 else None
    for orig_x, orig_y in positions_orig:
        db.garden.execute(
            """INSERT INTO plant_instances
               (instance_id, plot_id, plant_id, plant_type_id, variety_id,
                sow_style, row_id, location_x, location_y, spacing_inches,
                status, planned_sow_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [str(uuid4()), plot_id, variety_id or str(uuid4()),
             plant_type_id, variety_id, sow_style, row_id,
             orig_x, orig_y, spacing_inches, status, planned_sow_date],
        )
    db.garden.commit()


def _delete_instance(db: DatabaseConnections, instance_id: str) -> None:
    db.garden.execute("DELETE FROM harvest_log WHERE instance_id = ?", [instance_id])
    db.garden.execute("DELETE FROM plant_instances WHERE instance_id = ?", [instance_id])
    db.garden.commit()


def _delete_row(db: DatabaseConnections, row_id: str) -> None:
    rows = db.garden.execute(
        "SELECT instance_id FROM plant_instances WHERE row_id = ?", [row_id]
    ).fetchall()
    for (iid,) in rows:
        db.garden.execute("DELETE FROM harvest_log WHERE instance_id = ?", [iid])
    db.garden.execute("DELETE FROM plant_instances WHERE row_id = ?", [row_id])
    db.garden.commit()


def _ensure_amendment_table(db: DatabaseConnections) -> None:
    db.garden.execute("""
        CREATE TABLE IF NOT EXISTS soil_amendment_advice (
            advice_id       VARCHAR PRIMARY KEY,
            plot_id         VARCHAR NOT NULL,
            generated_at    TIMESTAMP NOT NULL,
            advice_text     VARCHAR NOT NULL
        )
    """)


def _load_latest_advice(db: DatabaseConnections, plot_id: str) -> dict | None:
    row = db.garden.execute(
        """SELECT advice_id, generated_at, advice_text
           FROM soil_amendment_advice
           WHERE plot_id = ?
           ORDER BY generated_at DESC LIMIT 1""",
        [plot_id],
    ).fetchone()
    if not row:
        return None
    return {"advice_id": row[0], "generated_at": row[1], "advice_text": row[2]}


def _save_advice(db: DatabaseConnections, plot_id: str, advice_text: str) -> None:
    db.garden.execute(
        """INSERT INTO soil_amendment_advice (advice_id, plot_id, generated_at, advice_text)
           VALUES (?, ?, ?, ?)""",
        [str(uuid4()), plot_id, datetime.now(timezone.utc), advice_text],
    )
    db.garden.commit()


# ── Garden maintenance helpers ────────────────────────────────────────────────

def _take_snapshot(db: DatabaseConnections) -> str:
    """
    Serialize current garden state into garden_snapshots.
    Marks all previous snapshots is_current=FALSE.

    Returns a human-readable result string.
    """
    garden_row = db.garden.execute(
        "SELECT garden_id, name, zipcode FROM garden LIMIT 1"
    ).fetchone()

    plots = db.garden.execute(
        "SELECT plot_id, name, area_sqft FROM plots ORDER BY name"
    ).fetchall()

    instances = db.garden.execute("""
        SELECT pi.instance_id, pi.plant_type_id, pi.plot_id,
               pi.status, pi.planted_date, pi.planned_sow_date,
               pi.harvest_count, pi.harvest_weight_lbs,
               pl.name AS plot_name
        FROM plant_instances pi
        JOIN plots pl ON pl.plot_id = pi.plot_id
        WHERE pi.status != 'archived'
        ORDER BY pi.plot_id, pi.plant_type_id
    """).fetchall()

    type_ids = list({r[1] for r in instances if r[1]})
    names_map: dict[str, str] = {}
    if type_ids:
        placeholders = ",".join("?" * len(type_ids))
        try:
            name_rows = db.plant.execute(
                f"SELECT plant_type_id, name FROM plant_types "
                f"WHERE plant_type_id IN ({placeholders})",
                type_ids,
            ).fetchall()
            names_map = {r[0]: r[1] for r in name_rows}
        except Exception:
            pass

    import json as _json
    snapshot = {
        "snapshot_date": str(date.today()),
        "garden": (
            {"garden_id": garden_row[0], "name": garden_row[1], "zipcode": garden_row[2]}
            if garden_row else None
        ),
        "plots": [{"plot_id": p[0], "name": p[1], "area_sqft": p[2]} for p in plots],
        "instances": [
            {
                "instance_id": r[0],
                "plant_type_id": r[1],
                "plant_name": names_map.get(r[1], r[1] or "unknown"),
                "plot_id": r[2],
                "plot_name": r[8],
                "status": r[3],
                "planted_date": str(r[4]) if r[4] else None,
                "planned_sow_date": str(r[5]) if r[5] else None,
                "harvest_count": r[6],
                "harvest_weight_lbs": r[7],
            }
            for r in instances
        ],
    }

    snapshot_id = str(uuid4())
    db.garden.execute("UPDATE garden_snapshots SET is_current = FALSE")
    db.garden.execute(
        """INSERT INTO garden_snapshots
               (snapshot_id, snapshot_date, triggered_by, snapshot_json, is_current)
           VALUES (?, ?, 'manual', ?, TRUE)""",
        [snapshot_id, date.today(), _json.dumps(snapshot)],
    )
    db.garden.commit()

    total = db.garden.execute("SELECT COUNT(*) FROM garden_snapshots").fetchone()[0]
    return (
        f"Snapshot saved — {len(instances)} plants across {len(plots)} plot(s). "
        f"{total} total snapshot(s) on record."
    )


def _run_health_check(db: DatabaseConnections) -> list[str]:
    """
    Scan garden state for actionable issues.

    Returns a list of human-readable alert strings (empty = all clear).
    """
    today = date.today()
    issues: list[str] = []

    # Overdue for harvest: active plants past their days_to_harvest window
    try:
        active_rows = db.garden.execute("""
            SELECT pi.plant_type_id, pi.planted_date, pl.name
            FROM plant_instances pi
            JOIN plots pl ON pl.plot_id = pi.plot_id
            WHERE pi.status = 'active' AND pi.planted_date IS NOT NULL
        """).fetchall()

        type_ids = list({r[0] for r in active_rows if r[0]})
        dth_map: dict[str, tuple[str, float]] = {}
        if type_ids:
            placeholders = ",".join("?" * len(type_ids))
            dth_rows = db.plant.execute(
                f"SELECT plant_type_id, name, AVG(days_to_harvest) "
                f"FROM plant_varieties "
                f"WHERE plant_type_id IN ({placeholders}) "
                f"GROUP BY plant_type_id, name",
                type_ids,
            ).fetchall()
            dth_map = {r[0]: (r[1], r[2]) for r in dth_rows if r[2]}

        for type_id, planted_date, plot_name in active_rows:
            if type_id in dth_map:
                plant_name, dth = dth_map[type_id]
                due = planted_date + timedelta(days=int(dth))
                overdue_days = (today - due).days
                if overdue_days > 0:
                    issues.append(
                        f"🍅 **{plant_name}** in {plot_name}: "
                        f"overdue for harvest by {overdue_days}d "
                        f"(planted {planted_date}, avg {int(dth)}d to harvest)"
                    )
    except Exception as exc:
        issues.append(f"⚠️ Error checking harvest overdue: {exc}")

    # Planned plants whose sow date has passed
    try:
        sow_rows = db.garden.execute("""
            SELECT pi.plant_type_id, pi.planned_sow_date, pl.name
            FROM plant_instances pi
            JOIN plots pl ON pl.plot_id = pi.plot_id
            WHERE pi.status = 'planned'
              AND pi.planned_sow_date IS NOT NULL
              AND pi.planned_sow_date < current_date
        """).fetchall()

        type_ids_s = list({r[0] for r in sow_rows if r[0]})
        names_s: dict[str, str] = {}
        if type_ids_s:
            placeholders = ",".join("?" * len(type_ids_s))
            name_rows = db.plant.execute(
                f"SELECT plant_type_id, name FROM plant_types "
                f"WHERE plant_type_id IN ({placeholders})",
                type_ids_s,
            ).fetchall()
            names_s = {r[0]: r[1] for r in name_rows}

        for type_id, sow_date, plot_name in sow_rows:
            plant_name = names_s.get(type_id, type_id or "Unknown")
            days_late = (today - sow_date).days
            issues.append(
                f"📅 **{plant_name}** in {plot_name}: "
                f"planned sow date was {sow_date} ({days_late}d ago) — still marked planned"
            )
    except Exception as exc:
        issues.append(f"⚠️ Error checking overdue sow dates: {exc}")

    # Active plants with no date at all (neither planted_date nor planned_sow_date)
    try:
        no_date_rows = db.garden.execute("""
            SELECT pi.plant_type_id, pl.name
            FROM plant_instances pi
            JOIN plots pl ON pl.plot_id = pi.plot_id
            WHERE pi.status = 'active'
              AND pi.planted_date IS NULL
              AND pi.planned_sow_date IS NULL
        """).fetchall()

        type_ids_n = list({r[0] for r in no_date_rows if r[0]})
        names_n: dict[str, str] = {}
        if type_ids_n:
            placeholders = ",".join("?" * len(type_ids_n))
            name_rows = db.plant.execute(
                f"SELECT plant_type_id, name FROM plant_types "
                f"WHERE plant_type_id IN ({placeholders})",
                type_ids_n,
            ).fetchall()
            names_n = {r[0]: r[1] for r in name_rows}

        for type_id, plot_name in no_date_rows:
            plant_name = names_n.get(type_id, type_id or "Unknown")
            issues.append(
                f"⚠️ **{plant_name}** in {plot_name}: "
                f"status is 'active' but no planted date recorded"
            )
    except Exception as exc:
        issues.append(f"⚠️ Error checking missing planted dates: {exc}")

    return issues


def _render_plant_maintenance(db: DatabaseConnections) -> None:
    """Snapshot and health check expander for the Plants tab."""
    with st.expander("🔧 Maintenance", expanded=False):
        col_s, col_h = st.columns(2)

        with col_s:
            if st.button("📸 Take Snapshot", use_container_width=True):
                with st.spinner("Saving snapshot…"):
                    msg = _take_snapshot(db)
                st.success(msg)

        with col_h:
            if st.button("🏥 Health Check", use_container_width=True):
                with st.spinner("Scanning garden…"):
                    issues = _run_health_check(db)
                if not issues:
                    st.success("All clear — no issues found.")
                else:
                    for issue in issues:
                        st.warning(issue)

        snap_rows = db.garden.execute(
            """SELECT snapshot_date, triggered_by
               FROM garden_snapshots
               ORDER BY snapshot_date DESC
               LIMIT 5"""
        ).fetchall()
        if snap_rows:
            total = db.garden.execute(
                "SELECT COUNT(*) FROM garden_snapshots"
            ).fetchone()[0]
            st.caption(f"{total} snapshot(s) on record:")
            for row in snap_rows:
                st.caption(f"  · {row[0]}  —  {row[1]}")


# ── PIL composite builder ─────────────────────────────────────────────────────

def _build_composite_bg(
    base_img: "Image.Image",
    poly_shapes: list[tuple[list, str, float]],
    instances: list[dict],
    crop_x: int, crop_y: int, crop_w: int, crop_h: int,
    canvas_w: int, canvas_h: int,
    px_per_foot: float | None,
    selected_id: str | None = None,
    selected_row_id: str | None = None,
) -> "Image.Image":
    """Composite region polygons and plant circles onto base_img using PIL."""
    poly_canvas = [
        (
            [cu.orig_to_canvas(p[0], p[1], crop_x, crop_y, crop_w, crop_h, canvas_w, canvas_h) for p in pts],
            color, alpha,
        )
        for pts, color, alpha in poly_shapes
    ]

    instance_data: list[dict] = []
    row_points: dict[str, list[tuple[float, float, str]]] = {}

    for inst in instances:
        if inst["location_x"] is None or inst["location_y"] is None:
            continue
        spacing = inst.get("spacing_inches") or 6.0
        radius = cu.spacing_to_canvas_px(spacing / 2, px_per_foot, crop_w, canvas_w)
        cx, cy = cu.orig_to_canvas(
            inst["location_x"], inst["location_y"],
            crop_x, crop_y, crop_w, crop_h, canvas_w, canvas_h,
        )
        color = cu.plant_type_color(inst["plant_type_id"])
        is_sel = inst["instance_id"] == selected_id or (
            selected_row_id is not None and inst.get("row_id") == selected_row_id
        )
        instance_data.append({
            "cx": cx, "cy": cy, "radius": radius,
            "color": color, "status": inst["status"], "selected": is_sel,
        })
        if inst.get("row_id"):
            row_points.setdefault(inst["row_id"], []).append((cx, cy, color))

    row_lines = [
        (pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], pts[i][2])
        for pts in row_points.values()
        for i in range(len(pts) - 1)
    ]

    return cu.composite_plant_canvas(base_img, poly_canvas, instance_data, row_lines)


def _find_nearest_instance(
    canvas_x: float,
    canvas_y: float,
    instances: list[dict],
    crop_x: int, crop_y: int, crop_w: int, crop_h: int,
    canvas_w: int, canvas_h: int,
) -> dict | None:
    """Return the closest plant instance within INSPECT_HIT_THRESHOLD_PX of the canvas click."""
    best: dict | None = None
    best_dist = INSPECT_HIT_THRESHOLD_PX
    for inst in instances:
        if inst["location_x"] is None:
            continue
        cx, cy = cu.orig_to_canvas(
            inst["location_x"], inst["location_y"],
            crop_x, crop_y, crop_w, crop_h, canvas_w, canvas_h,
        )
        dist = math.sqrt((canvas_x - cx) ** 2 + (canvas_y - cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = inst
    return best


# ── Plant details panel ───────────────────────────────────────────────────────

def _render_plant_details(
    db: DatabaseConnections,
    instance: dict,
    plant_types: list[dict],
) -> None:
    """Render the selected plant's info card with harvest log below the canvas."""
    name_map = {pt["plant_type_id"]: pt["name"] for pt in plant_types}
    plant_name = name_map.get(instance["plant_type_id"], "Unknown")
    varieties = _load_varieties(db, instance["plant_type_id"])
    variety = _load_variety_by_id(db, instance.get("variety_id") or "")

    st.divider()
    hdr_col, close_col = st.columns([11, 1])
    with hdr_col:
        variety_label = f" — {variety['variety_name']}" if variety else ""
        st.subheader(f"🌱 {plant_name}{variety_label}")
    with close_col:
        if st.button("✕", key="plant_detail_close", help="Close"):
            st.session_state.selected_plant_id = None
            st.rerun()

    days_to_harvest = (
        variety.get("days_to_harvest")
        if variety
        else _load_days_to_harvest(db, instance["plant_type_id"])
    )
    sow_date = instance.get("planted_date") or instance.get("planned_sow_date")
    if days_to_harvest and sow_date:
        harvest_ready = sow_date + timedelta(days=int(days_to_harvest))
        days_remaining = (harvest_ready - date.today()).days
        if days_remaining > 0:
            harvest_label = f"{harvest_ready} ({days_remaining}d)"
        elif days_remaining == 0:
            harvest_label = f"{harvest_ready} (today!)"
        else:
            harvest_label = f"{harvest_ready} ({abs(days_remaining)}d ago)"
    elif days_to_harvest:
        harvest_label = f"{days_to_harvest} days from sow"
    else:
        harvest_label = "—"

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Status", instance["status"])
    with col_b:
        _cur_sp = float(instance.get("spacing_inches") or 6.0)
        _new_sp = st.number_input(
            "Spacing (in)", min_value=1.0, max_value=72.0,
            value=_cur_sp, step=1.0,
            key=f"detail_spacing_{instance['instance_id']}",
        )
        if _new_sp != _cur_sp:
            _update_instance_spacing(db, instance["instance_id"], _new_sp)
            st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
            st.rerun()
    col_c.metric("Planned sow", str(instance.get("planned_sow_date") or "—"))
    col_d.metric("Harvest ready", harvest_label)

    # Variety picker
    if varieties:
        var_opts = {v["variety_name"]: v for v in varieties}
        var_names = ["(none)"] + list(var_opts.keys())
        cur_var_name = variety["variety_name"] if variety else "(none)"
        cur_var_idx = var_names.index(cur_var_name) if cur_var_name in var_names else 0
        new_var_name = st.selectbox(
            "Variety", var_names, index=cur_var_idx,
            key=f"detail_variety_{instance['instance_id']}",
        )
        if new_var_name != cur_var_name:
            if new_var_name == "(none)":
                new_sp = float(instance.get("spacing_inches") or 6.0)
                _update_instance_variety(db, instance["instance_id"], "", new_sp)
            else:
                v = var_opts[new_var_name]
                var_sp = v["spacing_inches"]
                new_sp = float(var_sp) if var_sp and var_sp > 0 else float(instance.get("spacing_inches") or 6.0)
                _update_instance_variety(db, instance["instance_id"], v["variety_id"], new_sp)
                # Sync the spacing input widget so it reflects the variety's value
                st.session_state[f"detail_spacing_{instance['instance_id']}"] = new_sp
            st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
            st.success("Variety updated.")
            st.rerun()
        if variety:
            st.caption(
                f"Height estimate: {variety.get('height_inches_estimate') or '—'} in  ·  "
                f"Sun tolerance: {variety.get('sun_tolerance') or '—'}"
            )

    status_opts = [s.value for s in garden_enum.PlantStatus]
    cur_idx = status_opts.index(instance["status"]) if instance["status"] in status_opts else 0

    act_col, status_col = st.columns([2, 3])
    with status_col:
        new_status = st.selectbox(
            "Change status", status_opts, index=cur_idx,
            key=f"detail_status_{instance['instance_id']}",
        )
        if new_status != instance["status"]:
            _update_instance_status(db, instance["instance_id"], new_status)
            st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
            st.rerun()

    with act_col:
        st.caption("Actions")
        row_id = instance.get("row_id")
        if row_id:
            if st.button("📦 Archive row", key=f"arch_row_{row_id}", use_container_width=True):
                _archive_row(db, row_id)
                st.session_state.selected_plant_id = None
                st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                st.rerun()
            if st.button("🗑️ Delete row", key=f"del_row_inspect_{row_id}", use_container_width=True, type="secondary"):
                _delete_row(db, row_id)
                st.session_state.selected_plant_id = None
                st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                st.rerun()
        else:
            if st.button("📦 Archive plant", key=f"arch_inst_{instance['instance_id']}", use_container_width=True):
                _archive_instance(db, instance["instance_id"])
                st.session_state.selected_plant_id = None
                st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                st.rerun()
            if st.button("🗑️ Delete plant", key=f"del_inst_inspect_{instance['instance_id']}", use_container_width=True, type="secondary"):
                _delete_instance(db, instance["instance_id"])
                st.session_state.selected_plant_id = None
                st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                st.rerun()

    # Harvest log — row members share a single combined log
    row_id = instance.get("row_id")
    st.divider()
    harvests = _load_row_harvest_log(db, row_id) if row_id else _load_harvest_log(db, instance["instance_id"])
    harvest_scope_label = "row harvest log" if row_id else "harvest log"

    total_qty = sum(h["quantity"] for h in harvests if h["quantity"]) if harvests else 0
    total_weight = sum(h["weight_lbs"] for h in harvests if h["weight_lbs"]) if harvests else 0.0

    hw1, hw2, hw3 = st.columns(3)
    hw1.metric("Harvest sessions", len(harvests))
    hw2.metric("Total count", total_qty if total_qty else "—")
    hw3.metric("Total weight", f"{total_weight:.2f} lbs" if total_weight else "—")

    if harvests:
        st.caption(f"History ({harvest_scope_label}):")
        for h in harvests:
            hc1, hc2, hc3, hc4, hc5 = st.columns([2, 1, 2, 4, 1])
            hc1.write(str(h["harvest_date"]))
            hc2.write(str(h["quantity"]) if h["quantity"] else "—")
            hc3.write(f"{h['weight_lbs']:.2f} lbs" if h["weight_lbs"] else "—")
            hc4.write(h["notes"] or "")
            with hc5:
                if st.button("🗑️", key=f"del_h_{h['harvest_id']}", help="Delete"):
                    _delete_harvest(db, h["harvest_id"], instance["instance_id"])
                    st.rerun()

    st.caption(f"Log a harvest ({harvest_scope_label}):")
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 2, 4, 1])
    with fc1:
        h_date = st.date_input(
            "Date", value=date.today(),
            key=f"h_date_{instance['instance_id']}", label_visibility="collapsed",
        )
    with fc2:
        h_qty = st.number_input(
            "Count", min_value=0, step=1, value=0,
            key=f"h_qty_{instance['instance_id']}", label_visibility="collapsed",
            help="Number of items harvested",
        )
    with fc3:
        h_weight = st.number_input(
            "Weight (lbs)", min_value=0.0, step=0.1, value=0.0, format="%.2f",
            key=f"h_weight_{instance['instance_id']}", label_visibility="collapsed",
        )
    with fc4:
        h_notes = st.text_input(
            "Notes", value="", placeholder="Notes (optional)",
            key=f"h_notes_{instance['instance_id']}", label_visibility="collapsed",
        )
    with fc5:
        if st.button("➕ Add", key=f"add_h_{instance['instance_id']}"):
            _add_harvest(
                db, instance["instance_id"], h_date,
                h_qty if h_qty > 0 else None,
                h_weight if h_weight > 0 else None,
                h_notes or None,
            )
            st.success("Harvest logged.")
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def render(db: DatabaseConnections) -> None:
    _ensure_migrations(db)

    # Force one rerun on first visit so fabric.js initialises its background
    # image correctly after being rendered inside a hidden Streamlit tab.
    # Bump plants_cv so the canvas key is fresh (never seen in the broken hidden-tab render).
    if "plants_tab_ready" not in st.session_state:
        st.session_state.plants_tab_ready = True
        st.session_state.plants_cv = 1
        st.rerun()

    garden = _load_garden(db)
    if not garden:
        st.info("Set up your garden on the Map tab first.")
        return

    image_path = Path(garden["aerial_image_path"]) if garden["aerial_image_path"] else None
    if not image_path or not image_path.exists():
        st.warning("Aerial image not found — re-upload on the Map tab.")
        return

    plots = _load_plots(db, garden["garden_id"])
    if not plots:
        st.info("No plots yet — draw plots on the Map tab first.")
        return

    plant_types = _load_plant_types(db)
    if not plant_types:
        st.info("No plant data found — run the plant_db_team build first.")
        return

    img = Image.open(str(image_path))
    px_per_foot = garden.get("scale_px_per_foot")

    if "plants_cv" not in st.session_state:
        st.session_state.plants_cv = 0
    if "selected_plant_id" not in st.session_state:
        st.session_state.selected_plant_id = None

    # ── Config row ─────────────────────────────────────────────────────────────
    st.subheader("🌿 Plants")
    mode_col, c1, c2, c3, c_sp, c4, c5, c6 = st.columns([1.5, 2, 2, 1, 1, 1, 2, 1])

    with mode_col:
        plants_mode = st.radio(
            "Mode", ["✏️ Place", "🔍 Inspect"],
            index=1,
            key="plants_mode",
        )
    with c1:
        selected_plot_name = st.selectbox("Plot", [p["name"] for p in plots], key="plants_plot_select")
    with c2:
        type_options = {f"{pt['name']} ({pt['category']})": pt for pt in plant_types}
        selected_type_label = st.selectbox("Plant type", list(type_options.keys()), key="plants_type_select")
        selected_type = type_options[selected_type_label]
    with c3:
        sow_style = st.radio("Sow style", ["Single", "Row"], key="plants_sow_style")

    # Auto-suggest spacing from the selected plant type's variety data.
    # Uses a per-type session key so switching types resets to that type's suggested value.
    _hint_vars = _load_varieties(db, selected_type["plant_type_id"])
    _hint_sp = [v["spacing_inches"] for v in _hint_vars if v.get("spacing_inches") and v["spacing_inches"] > 0]
    _suggested_sp = round(sum(_hint_sp) / len(_hint_sp)) if _hint_sp else 6
    _sp_key = f"plants_spacing_{selected_type['plant_type_id']}"
    if _sp_key not in st.session_state:
        st.session_state[_sp_key] = _suggested_sp

    with c_sp:
        spacing_inches = st.number_input(
            "Spacing (in)", min_value=1, max_value=72,
            step=1, key=_sp_key,
            help="Plant spacing in inches — auto-suggested from variety data",
        )

    with c4:
        status = st.selectbox("Status", [s.value for s in garden_enum.PlantStatus], key="plants_status_select")
    with c5:
        planned_date = st.date_input("Planned sow date", value=None, key="plants_sow_date")
    with c6:
        st.caption("Layers")
        show_sun = st.checkbox("☀️ Sun", value=False, key="plants_show_sun")
        show_height = st.checkbox("📐 Height", value=False, key="plants_show_height")

    if not px_per_foot:
        st.caption("⚠️ No scale set on Map tab — plant spacing is approximate.")

    # ── Canvas ─────────────────────────────────────────────────────────────────
    plot = next(p for p in plots if p["name"] == selected_plot_name)
    plot_full = _load_plot_full(db, plot["plot_id"])
    if not plot_full:
        st.error("Could not load plot.")
        return

    plot_pts = json.loads(plot_full["polygon"]) if isinstance(plot_full["polygon"], str) else plot_full["polygon"]
    canvas_width = DEFAULT_CANVAS_WIDTH
    crop_img, crop_x, crop_y, crop_w, crop_h = cu.get_plot_crop(img, plot_pts, canvas_width)
    canvas_height = crop_img.height

    regions = _load_regions(db, plot["plot_id"])
    instances = _load_instances(db, plot["plot_id"])

    poly_shapes = [(plot_pts, cu.PLOT_COLORS[0], 0.15)]
    for r in regions:
        if r["region_type"] == "sun" and not show_sun:
            continue
        if r["region_type"] == "height" and not show_height:
            continue
        pts = json.loads(r["polygon"]) if isinstance(r["polygon"], str) else r["polygon"]
        color_map = cu.SUN_ZONE_COLORS if r["region_type"] == "sun" else cu.HEIGHT_ZONE_COLORS
        poly_shapes.append((pts, color_map.get(r["zone_value"], "#888888"), cu.REGION_ALPHA))

    sel_id = st.session_state.selected_plant_id
    sel_inst = next((i for i in instances if i["instance_id"] == sel_id), None) if sel_id else None
    sel_row_id = sel_inst.get("row_id") if sel_inst else None

    # Composite everything onto the background image — avoids fabric.js initial_drawing
    # failing to initialize when the canvas is inside a hidden Streamlit tab.
    bg_img = _build_composite_bg(
        crop_img, poly_shapes, instances,
        crop_x, crop_y, crop_w, crop_h, canvas_width, canvas_height, px_per_foot,
        selected_id=sel_id, selected_row_id=sel_row_id,
    )

    is_inspect = plants_mode == "🔍 Inspect"
    if is_inspect:
        drawing_mode = "point"
        point_r = INSPECT_POINT_RADIUS
        st.caption("Click a plant circle to inspect it.")
    else:
        radius_canvas = cu.spacing_to_canvas_px(spacing_inches / 2, px_per_foot, crop_w, canvas_width)
        point_r = max(4, int(radius_canvas))
        drawing_mode = "point" if sow_style == "Single" else "line"
        st.caption(
            "Click to place a plant." if sow_style == "Single"
            else "Draw a line to place a row — plants auto-spaced."
        )

    canvas_result = st_canvas(
        fill_color=cu.hex_to_rgba(cu.PLANT_STATUS_COLORS.get(status, "#3cb44b"), 0.5),
        stroke_color=cu.PLANT_STATUS_COLORS.get(status, "#3cb44b"),
        stroke_width=2,
        background_image=bg_img,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        point_display_radius=point_r,
        display_toolbar=False,
        initial_drawing={"version": "4.4.0", "objects": []},
        key=f"plants_canvas_{plot['plot_id']}_{sow_style}_{selected_type['plant_type_id']}_{plants_mode}_{st.session_state.plants_cv}",
    )

    if canvas_result.json_data:
        new_objects = canvas_result.json_data.get("objects", [])

        if is_inspect:
            new_circles = [o for o in new_objects if o.get("type") == "circle"]
            if new_circles:
                obj = new_circles[-1]
                r = obj.get("radius", INSPECT_POINT_RADIUS)
                click_cx = obj.get("left", 0) + r
                click_cy = obj.get("top", 0) + r
                hit = _find_nearest_instance(
                    click_cx, click_cy, instances,
                    crop_x, crop_y, crop_w, crop_h, canvas_width, canvas_height,
                )
                st.session_state.selected_plant_id = hit["instance_id"] if hit else None
                st.session_state.plants_cv += 1
                st.rerun()

        elif sow_style == "Single":
            new_circles = [o for o in new_objects if o.get("type") == "circle"]
            if new_circles:
                obj = new_circles[-1]
                r = obj.get("radius", point_r)
                cx, cy = obj.get("left", 0) + r, obj.get("top", 0) + r
                orig_x, orig_y = cu.canvas_to_orig(cx, cy, crop_x, crop_y, crop_w, crop_h, canvas_width, canvas_height)
                _save_instances(
                    db, plot["plot_id"], selected_type["plant_type_id"],
                    "",
                    "single", status, planned_date or None, [(orig_x, orig_y)], spacing_inches,
                )
                st.session_state.plants_cv += 1
                st.rerun()

        else:
            new_lines = [o for o in new_objects if o.get("type") == "line"]
            if new_lines:
                obj = new_lines[-1]
                x1 = obj.get("x1", 0) + obj.get("left", 0)
                y1 = obj.get("y1", 0) + obj.get("top", 0)
                x2 = obj.get("x2", 0) + obj.get("left", 0)
                y2 = obj.get("y2", 0) + obj.get("top", 0)
                spacing_px = cu.spacing_to_canvas_px(spacing_inches, px_per_foot, crop_w, canvas_width)
                canvas_positions = cu.row_plant_positions(x1, y1, x2, y2, spacing_px)
                orig_positions = [
                    cu.canvas_to_orig(cx, cy, crop_x, crop_y, crop_w, crop_h, canvas_width, canvas_height)
                    for cx, cy in canvas_positions
                ]
                _save_instances(
                    db, plot["plot_id"], selected_type["plant_type_id"],
                    "",
                    "row", status, planned_date or None, orig_positions, spacing_inches,
                )
                n = len(orig_positions)
                st.session_state.plants_cv += 1
                st.success(f"Row of {n} plant{'s' if n != 1 else ''} saved.")
                st.rerun()

    # Show selected plant details immediately below canvas
    if st.session_state.selected_plant_id:
        selected_inst = next(
            (i for i in instances if i["instance_id"] == st.session_state.selected_plant_id),
            None,
        )
        if selected_inst:
            _render_plant_details(db, selected_inst, plant_types)
        else:
            st.session_state.selected_plant_id = None

    # ── Maintenance ───────────────────────────────────────────────────────────
    _render_plant_maintenance(db)

    # ── Soil Amendment Advisor ────────────────────────────────────────────────
    _render_amendment_advisor(db, plot["plot_id"], plot["name"])

    # ── Layout Optimizer ──────────────────────────────────────────────────────
    layout_optimizer.render_section(db, plot["plot_id"], plot["name"])

    # ── Plant list ────────────────────────────────────────────────────────────
    _render_plant_list(db, plot["plot_id"], plant_types)


# ── Soil Amendment Advisor ────────────────────────────────────────────────────

def _render_amendment_advisor(
    db: DatabaseConnections,
    plot_id: str,
    plot_name: str,
) -> None:
    """Render the soil amendment advisor section for the selected plot."""
    _ensure_amendment_table(db)

    st.divider()
    with st.expander("🧪 Soil Amendment Advisor", expanded=False):
        prior = _load_latest_advice(db, plot_id)
        if prior:
            st.caption(f"Last generated: {prior['generated_at']}")
            st.markdown(prior["advice_text"])
            regen_label = "🔄 Regenerate advice"
        else:
            st.caption("No advice generated yet for this plot.")
            regen_label = "🧪 Get soil amendment advice"

        if st.button(regen_label, key=f"amend_run_{plot_id}"):
            with st.spinner("Analyzing soil needs and generating recommendations…"):
                try:
                    advice = asyncio.run(
                        amendment_agent.run(plot_id, plot_name, db.garden, db.plant)
                    )
                    _save_advice(db, plot_id, advice)
                    st.success("Amendment advice ready!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Agent error: {exc}")


# ── Plant list ────────────────────────────────────────────────────────────────

def _render_plant_list(
    db: DatabaseConnections,
    plot_id: str,
    plant_types: list[dict],
) -> None:
    instances = _load_instances(db, plot_id)
    archived_count = _count_archived(db, plot_id)
    if not instances:
        st.caption("No plants placed yet — click the map to add some.")
        if archived_count:
            st.caption(f"📦 {archived_count} archived plants in this plot (hidden from view, kept in DB).")
        return

    name_map = {pt["plant_type_id"]: pt["name"] for pt in plant_types}

    st.divider()

    # ── Date range filter ──────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        date_from = st.date_input(
            "Planted from", value=None,
            key=f"list_date_from_{plot_id}", label_visibility="collapsed",
            help="Filter by planted / planned sow date — from",
        )
    with fc2:
        date_to = st.date_input(
            "Planted to", value=None,
            key=f"list_date_to_{plot_id}", label_visibility="collapsed",
            help="Filter by planted / planned sow date — to",
        )
    with fc3:
        if st.button("✕ Clear", key=f"list_date_clear_{plot_id}", help="Clear date filter"):
            st.session_state[f"list_date_from_{plot_id}"] = None
            st.session_state[f"list_date_to_{plot_id}"] = None
            st.rerun()

    if date_from or date_to:
        filtered = []
        for inst in instances:
            inst_date = inst.get("planted_date") or inst.get("planned_sow_date")
            if inst_date is None:
                filtered.append(inst)
                continue
            if date_from and inst_date < date_from:
                continue
            if date_to and inst_date > date_to:
                continue
            filtered.append(inst)
        instances = filtered

    archive_note = f" · 📦 {archived_count} archived" if archived_count else ""
    filter_note  = f" · filtered {len(instances)}" if (date_from or date_to) else ""
    st.subheader(f"Plants in plot ({len(instances)}{filter_note}{archive_note})")

    shown_rows: set[str] = set()
    for inst in instances:
        plant_name = name_map.get(inst["plant_type_id"], "Unknown")
        rid = inst.get("row_id")

        if rid and rid in shown_rows:
            continue

        if rid:
            row_insts = [i for i in instances if i.get("row_id") == rid]
            label = f"⬛ {plant_name} — row of {len(row_insts)} · {inst['status']}"
            with st.expander(label):
                col_a, col_b = st.columns(2)
                col_a.caption(f"Sow date: {inst['planned_sow_date'] or '—'}")
                col_b.caption(f"Spacing: {inst['spacing_inches'] or '—'} in")
                if st.button("🗑️ Delete row", key=f"del_row_{rid}", type="secondary"):
                    _delete_row(db, rid)
                    st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                    st.rerun()
            shown_rows.add(rid)
        else:
            label = f"🌱 {plant_name} — {inst['status']}"
            with st.expander(label):
                col_a, col_b = st.columns(2)
                col_a.caption(f"Sow date: {inst['planned_sow_date'] or '—'}")
                col_b.caption(f"Spacing: {inst['spacing_inches'] or '—'} in")
                if st.button("🗑️ Delete", key=f"del_inst_{inst['instance_id']}", type="secondary"):
                    _delete_instance(db, inst["instance_id"])
                    st.session_state.plants_cv = st.session_state.get("plants_cv", 0) + 1
                    st.rerun()
