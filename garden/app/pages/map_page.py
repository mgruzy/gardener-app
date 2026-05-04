"""Map page — garden setup, aerial image upload, plot polygon builder, and sub-regions."""

import json
import math
from datetime import date
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import app.utils.canvas_utils  # applies the image_to_url compatibility shim at import

from app.db.connections import DatabaseConnections
import app.internal_objects.garden_types as garden_enum


# --- Parameters ---
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
AERIAL_IMAGE_DIR = "aerial"
DEFAULT_CANVAS_WIDTH = 700
PLOT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]
SUN_ZONE_COLORS = {
    "full_sun":      "#FFD700",
    "partial_shade": "#4682B4",
    "full_shade":    "#191970",
}
HEIGHT_ZONE_COLORS = {
    "low":    "#90EE90",
    "medium": "#228B22",
    "tall":   "#006400",
}
REGION_ALPHA = 0.35


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_garden(db: DatabaseConnections) -> dict | None:
    row = db.garden.execute(
        "SELECT garden_id, name, zipcode, aerial_image_path, scale_px_per_foot FROM garden LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return dict(zip(["garden_id", "name", "zipcode", "aerial_image_path", "scale_px_per_foot"], row))


def _save_garden(db: DatabaseConnections, name: str, zipcode: str, image_path: str) -> str:
    garden_id = str(uuid4())
    db.garden.execute(
        "INSERT INTO garden (garden_id, name, zipcode, aerial_image_path, created_at) VALUES (?, ?, ?, ?, ?)",
        [garden_id, name, zipcode, image_path, date.today()],
    )
    return garden_id


def _save_scale(db: DatabaseConnections, garden_id: str, px_per_foot: float) -> None:
    db.garden.execute(
        "UPDATE garden SET scale_px_per_foot = ? WHERE garden_id = ?",
        [px_per_foot, garden_id],
    )


def _load_plots(db: DatabaseConnections, garden_id: str) -> list[dict]:
    rows = db.garden.execute(
        "SELECT plot_id, name, area_sqft, sun_zone_default, polygon FROM plots WHERE garden_id = ? ORDER BY name",
        [garden_id],
    ).fetchall()
    return [dict(zip(["plot_id", "name", "area_sqft", "sun_zone_default", "polygon"], r)) for r in rows]


def _save_plot(
    db: DatabaseConnections,
    garden_id: str,
    name: str,
    polygon: list,
    sun_zone: str,
    height_zone: str,
    canvas_width: int,
    canvas_height: int,
    image_width: int,
    image_height: int,
    px_per_foot: float | None,
) -> None:
    scale_x = image_width / canvas_width
    scale_y = image_height / canvas_height
    scaled = [[pt[0] * scale_x, pt[1] * scale_y] for pt in polygon]
    area = _compute_area(scaled, px_per_foot)
    db.garden.execute(
        """INSERT INTO plots (plot_id, garden_id, name, polygon, area_sqft, sun_zone_default, height_zone_default)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [str(uuid4()), garden_id, name, json.dumps(scaled), round(area, 1), sun_zone, height_zone],
    )


def _delete_plot(db: DatabaseConnections, plot_id: str) -> None:
    db.garden.execute("DELETE FROM plot_regions WHERE plot_id = ?", [plot_id])
    db.garden.execute("DELETE FROM plots WHERE plot_id = ?", [plot_id])


def _load_regions(db: DatabaseConnections, plot_id: str) -> list[dict]:
    rows = db.garden.execute(
        """SELECT region_id, region_type, zone_value, polygon, area_sqft
           FROM plot_regions WHERE plot_id = ? ORDER BY region_type, zone_value""",
        [plot_id],
    ).fetchall()
    return [dict(zip(["region_id", "region_type", "zone_value", "polygon", "area_sqft"], r)) for r in rows]


def _save_region(
    db: DatabaseConnections,
    plot_id: str,
    region_type: str,
    zone_value: str,
    polygon: list,
    canvas_width: int,
    canvas_height: int,
    shown_w: int,
    shown_h: int,
    px_per_foot: float | None,
    crop_x: int = 0,
    crop_y: int = 0,
) -> None:
    """Save region polygon. shown_w/shown_h are the dimensions of the image slice displayed
    (full image or a crop). crop_x/crop_y are the top-left of that slice in original image space."""
    scale_x = shown_w / canvas_width
    scale_y = shown_h / canvas_height
    scaled = [[pt[0] * scale_x + crop_x, pt[1] * scale_y + crop_y] for pt in polygon]
    area = _compute_area(scaled, px_per_foot)
    db.garden.execute(
        """INSERT INTO plot_regions (region_id, plot_id, region_type, zone_value, polygon, area_sqft)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [str(uuid4()), plot_id, region_type, zone_value, json.dumps(scaled), round(area, 1)],
    )


def _delete_region(db: DatabaseConnections, region_id: str) -> None:
    db.garden.execute("DELETE FROM plot_regions WHERE region_id = ?", [region_id])


def _delete_all_regions(db: DatabaseConnections, plot_id: str) -> None:
    db.garden.execute("DELETE FROM plot_regions WHERE plot_id = ?", [plot_id])


# ── Geometry ──────────────────────────────────────────────────────────────────

def _shoelace_area(polygon: list) -> float:
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _compute_area(polygon_img_space: list, px_per_foot: float | None) -> float:
    px_area = _shoelace_area(polygon_img_space)
    if px_per_foot and px_per_foot > 0:
        return px_area / (px_per_foot ** 2)
    return px_area / (96 * 12) ** 2  # rough fallback at 96dpi


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_canvas_path(obj: dict) -> list[list[float]]:
    """Extract [x, y] points from a drawable-canvas path object (list-of-command format)."""
    path = obj.get("path", [])
    if not path:
        return []
    points = []
    for cmd in path:
        if not isinstance(cmd, list) or len(cmd) < 3:
            continue
        if cmd[0] in ("M", "L"):
            try:
                points.append([float(cmd[1]), float(cmd[2])])
            except (ValueError, TypeError):
                pass
    return points


def _plots_to_canvas_shapes(
    plots: list[dict], orig_w: int, orig_h: int, canvas_w: int, canvas_h: int
) -> list[dict]:
    shapes = []
    scale_x = canvas_w / orig_w
    scale_y = canvas_h / orig_h
    for i, plot in enumerate(plots):
        try:
            pts = json.loads(plot["polygon"]) if isinstance(plot["polygon"], str) else plot["polygon"]
        except Exception:
            continue
        if not pts:
            continue
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        path = "M " + " L ".join(f"{pt[0]*scale_x:.1f} {pt[1]*scale_y:.1f}" for pt in pts) + " Z"
        shapes.append({
            "type": "path", "path": path,
            "stroke": color, "fill": _hex_to_rgba(color, 0.15),
            "strokeWidth": 2, "selectable": False,
        })
    return shapes


def _get_plot_crop(
    img: Image.Image,
    polygon_orig: list,
    canvas_width: int,
    padding_frac: float = 0.25,
) -> tuple[Image.Image, int, int, int, int]:
    """Crop original image to the plot bounding box with padding.

    Returns (cropped_img_at_canvas_width, crop_x1, crop_y1, crop_w, crop_h).
    All crop values are in original image pixel space.
    """
    xs = [pt[0] for pt in polygon_orig]
    ys = [pt[1] for pt in polygon_orig]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    cx1 = max(0, int(bx1) - pad_x)
    cy1 = max(0, int(by1) - pad_y)
    cx2 = min(img.width, int(bx2) + pad_x)
    cy2 = min(img.height, int(by2) + pad_y)
    cw, ch = cx2 - cx1, cy2 - cy1
    cropped = img.crop((cx1, cy1, cx2, cy2))
    canvas_h = int(ch * canvas_width / cw)
    return cropped.resize((canvas_width, canvas_h)), cx1, cy1, cw, ch


def _shapes_for_crop(
    polygons_orig: list[tuple[list, str, float]],
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    canvas_w: int,
    canvas_h: int,
) -> list[dict]:
    """Convert polygons in original image space to canvas shapes for a cropped view.

    polygons_orig: list of (pts, stroke_color, fill_alpha) tuples.
    """
    scale_x = canvas_w / crop_w
    scale_y = canvas_h / crop_h
    shapes = []
    for pts, color, alpha in polygons_orig:
        if not pts:
            continue
        path = "M " + " L ".join(
            f"{(pt[0] - crop_x) * scale_x:.1f} {(pt[1] - crop_y) * scale_y:.1f}"
            for pt in pts
        ) + " Z"
        shapes.append({
            "type": "path", "path": path,
            "stroke": color, "fill": _hex_to_rgba(color, alpha),
            "strokeWidth": 2, "selectable": False,
        })
    return shapes


def _regions_to_canvas_shapes(
    regions: list[dict], orig_w: int, orig_h: int, canvas_w: int, canvas_h: int
) -> list[dict]:
    shapes = []
    scale_x = canvas_w / orig_w
    scale_y = canvas_h / orig_h
    for region in regions:
        try:
            pts = json.loads(region["polygon"]) if isinstance(region["polygon"], str) else region["polygon"]
        except Exception:
            continue
        if not pts:
            continue
        color_map = SUN_ZONE_COLORS if region["region_type"] == "sun" else HEIGHT_ZONE_COLORS
        color = color_map.get(region["zone_value"], "#888888")
        path = "M " + " L ".join(f"{pt[0]*scale_x:.1f} {pt[1]*scale_y:.1f}" for pt in pts) + " Z"
        shapes.append({
            "type": "path", "path": path,
            "stroke": color, "fill": _hex_to_rgba(color, REGION_ALPHA),
            "strokeWidth": 2, "selectable": False,
        })
    return shapes


# ── Schema migrations ─────────────────────────────────────────────────────────

def _ensure_schema_migrations(db: DatabaseConnections) -> None:
    db.garden.execute("ALTER TABLE garden ADD COLUMN IF NOT EXISTS scale_px_per_foot DOUBLE")
    db.garden.execute("""
        CREATE TABLE IF NOT EXISTS plot_regions (
            region_id   VARCHAR PRIMARY KEY,
            plot_id     VARCHAR NOT NULL,
            region_type VARCHAR NOT NULL,
            zone_value  VARCHAR NOT NULL,
            polygon     JSON NOT NULL,
            area_sqft   DOUBLE
        )
    """)


# ── Entry point ───────────────────────────────────────────────────────────────

def render(db: DatabaseConnections, data_dir: Path) -> None:
    image_dir = data_dir / "images" / AERIAL_IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    _ensure_schema_migrations(db)

    garden = _load_garden(db)
    if garden is None:
        _render_setup(db, image_dir)
    else:
        _render_garden(db, garden, image_dir)


# ── Garden setup ──────────────────────────────────────────────────────────────

def _render_setup(db: DatabaseConnections, image_dir: Path) -> None:
    st.subheader("Set up your garden")
    st.caption("You only need to do this once.")

    with st.form("garden_setup"):
        name = st.text_input("Garden name", placeholder="e.g. Backyard")
        zipcode = st.text_input("Zipcode", placeholder="e.g. 98115")
        image_file = st.file_uploader("Aerial image", type=SUPPORTED_IMAGE_TYPES)
        submitted = st.form_submit_button("Create garden", type="primary")

    if submitted:
        if not name or not zipcode:
            st.error("Garden name and zipcode are required.")
            return
        if not image_file:
            st.error("Please upload an aerial image.")
            return
        suffix = Path(image_file.name).suffix
        image_path = image_dir / f"aerial{suffix}"
        image_path.write_bytes(image_file.read())
        _save_garden(db, name.strip(), zipcode.strip(), str(image_path))
        st.success(f"Garden '{name}' created!")
        st.rerun()


# ── Garden main view ──────────────────────────────────────────────────────────

def _render_garden(db: DatabaseConnections, garden: dict, image_dir: Path) -> None:
    col_info, col_meta = st.columns([3, 1])
    with col_info:
        st.subheader(garden["name"])
    with col_meta:
        px_per_foot = garden.get("scale_px_per_foot")
        scale_label = f"  ·  {px_per_foot:.1f} px/ft ✓" if px_per_foot else "  ·  no scale"
        st.caption(f"📍 {garden['zipcode']}{scale_label}")

    image_path = Path(garden["aerial_image_path"]) if garden["aerial_image_path"] else None
    if not image_path or not image_path.exists():
        st.warning("Aerial image not found.")
        return

    img = Image.open(str(image_path))
    orig_w, orig_h = img.size

    canvas_width = st.slider("Canvas width (px)", 300, 1400, DEFAULT_CANVAS_WIDTH, 50)
    canvas_height = int(orig_h * canvas_width / orig_w)
    img_resized = img.resize((canvas_width, canvas_height))

    plots = _load_plots(db, garden["garden_id"])

    mode = st.radio(
        "Mode",
        ["✏️ Draw plot", "👁️ View plots", "📏 Set scale", "🔍 Plot regions"],
        horizontal=True,
    )

    if mode == "✏️ Draw plot":
        _render_draw_mode(db, garden, img_resized, plots, canvas_width, canvas_height, orig_w, orig_h)
    elif mode == "👁️ View plots":
        _render_view_mode(db, img_resized, plots, canvas_width, canvas_height, orig_w, orig_h)
    elif mode == "📏 Set scale":
        _render_scale_mode(db, garden, img_resized, canvas_width, canvas_height)
    else:
        _render_regions_mode(db, garden, img, plots, canvas_width, orig_w, orig_h)


# ── Draw mode ─────────────────────────────────────────────────────────────────

def _render_draw_mode(
    db: DatabaseConnections,
    garden: dict,
    img_resized: Image.Image,
    plots: list[dict],
    canvas_width: int,
    canvas_height: int,
    orig_w: int,
    orig_h: int,
) -> None:
    st.caption("Click to place points. Right-click to finish the polygon.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.15)",
        stroke_width=2,
        stroke_color="#ff6600",
        background_image=img_resized,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        display_toolbar=False,
        key=f"plot_canvas_{canvas_width}",
    )

    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        paths = [obj for obj in objects if obj.get("type") == "path"]
        if paths:
            _render_save_plot_form(db, garden, paths[-1], canvas_width, canvas_height, orig_w, orig_h)

    _render_plot_list(db, plots)


# ── View mode ─────────────────────────────────────────────────────────────────

def _render_view_mode(
    db: DatabaseConnections,
    img_resized: Image.Image,
    plots: list[dict],
    canvas_width: int,
    canvas_height: int,
    orig_w: int,
    orig_h: int,
) -> None:
    if not plots:
        st.image(img_resized, width=canvas_width)
        st.info("No plots yet — switch to Draw mode to add one.")
        return

    shapes = _plots_to_canvas_shapes(plots, orig_w, orig_h, canvas_width, canvas_height)
    st_canvas(
        fill_color="rgba(255, 165, 0, 0.15)",
        stroke_width=2,
        stroke_color="#ff6600",
        background_image=img_resized,
        update_streamlit=False,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="transform",
        display_toolbar=False,
        initial_drawing={"version": "4.4.0", "objects": shapes},
        key=f"plot_view_canvas_{canvas_width}",
    )
    _render_plot_list(db, plots)


# ── Scale mode ────────────────────────────────────────────────────────────────

def _render_scale_mode(
    db: DatabaseConnections,
    garden: dict,
    img_resized: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> None:
    px_per_foot = garden.get("scale_px_per_foot")
    line_key = f"scale_line_{canvas_width}"

    if px_per_foot:
        col_info, col_clear = st.columns([5, 1])
        with col_info:
            st.info(f"Current scale: **{px_per_foot:.1f} px/ft** — areas will be accurate.")
        with col_clear:
            if st.button("🗑️ Clear scale", type="secondary"):
                db.garden.execute(
                    "UPDATE garden SET scale_px_per_foot = NULL WHERE garden_id = ?",
                    [garden["garden_id"]],
                )
                st.session_state.pop(line_key, None)
                st.rerun()
    else:
        st.warning("No scale set — area estimates are approximate.")

    st.caption("Draw a line along something you know the real-world length of (a fence, a wall, a known distance).")

    # Persist the drawn line across reruns via session state
    stored_line = st.session_state.get(line_key)
    initial_drawing = {"version": "4.4.0", "objects": [stored_line]} if stored_line else None

    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#00BFFF",
        fill_color="rgba(0,191,255,0.1)",
        background_image=img_resized,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="line",
        display_toolbar=False,
        initial_drawing=initial_drawing,
        key=f"scale_canvas_{canvas_width}",
    )

    line_px = None
    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        lines = [obj for obj in objects if obj.get("type") == "line"]
        if lines:
            line_obj = lines[-1]
            st.session_state[line_key] = line_obj
            x1 = line_obj.get("x1", 0) + line_obj.get("left", 0)
            y1 = line_obj.get("y1", 0) + line_obj.get("top", 0)
            x2 = line_obj.get("x2", 0) + line_obj.get("left", 0)
            y2 = line_obj.get("y2", 0) + line_obj.get("top", 0)
            line_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            st.caption(f"Line drawn: **{line_px:.1f} px**")

    if line_px and line_px > 1:
        with st.form("set_scale"):
            real_ft = st.number_input("Real-world length of this line (feet)", min_value=0.1, value=10.0, step=0.5)
            save = st.form_submit_button("Set scale", type="primary")

        if save:
            _save_scale(db, garden["garden_id"], line_px / real_ft)
            st.success(f"Scale set: {line_px / real_ft:.1f} px/ft")
            st.rerun()


# ── Regions mode ──────────────────────────────────────────────────────────────

def _render_regions_mode(
    db: DatabaseConnections,
    garden: dict,
    img: Image.Image,
    plots: list[dict],
    canvas_width: int,
    orig_w: int,
    orig_h: int,
) -> None:
    if not plots:
        st.info("No plots yet — draw plots first.")
        return

    col_plot, col_type = st.columns(2)
    with col_plot:
        selected_name = st.selectbox("Plot", [p["name"] for p in plots])
    with col_type:
        region_type_label = st.radio("Region type", ["☀️ Sun zone", "📐 Height zone"], horizontal=True)

    plot = next(p for p in plots if p["name"] == selected_name)
    is_sun = region_type_label == "☀️ Sun zone"
    region_type = "sun" if is_sun else "height"

    zone_options = [z.value for z in garden_enum.SunZone] if is_sun else [z.value for z in garden_enum.HeightZone]
    zone_colors = SUN_ZONE_COLORS if is_sun else HEIGHT_ZONE_COLORS
    selected_zone = st.selectbox("Zone value to draw", zone_options)
    stroke_color = zone_colors.get(selected_zone, "#888888")

    regions = _load_regions(db, plot["plot_id"])

    # Crop the image to this plot's bounding box
    plot_pts = json.loads(plot["polygon"]) if isinstance(plot["polygon"], str) else plot["polygon"]
    crop_img, crop_x, crop_y, crop_w, crop_h = _get_plot_crop(img, plot_pts, canvas_width)
    canvas_height = crop_img.height

    # Build initial shapes in cropped canvas space
    all_plots_in_view = []
    for i, p in enumerate(plots):
        pts = json.loads(p["polygon"]) if isinstance(p["polygon"], str) else p["polygon"]
        color = PLOT_COLORS[plots.index(p) % len(PLOT_COLORS)]
        alpha = 0.3 if p["plot_id"] == plot["plot_id"] else 0.1
        all_plots_in_view.append((pts, color, alpha))

    region_entries = [
        (
            json.loads(r["polygon"]) if isinstance(r["polygon"], str) else r["polygon"],
            (SUN_ZONE_COLORS if r["region_type"] == "sun" else HEIGHT_ZONE_COLORS).get(r["zone_value"], "#888888"),
            REGION_ALPHA,
        )
        for r in regions
    ]

    initial_shapes = _shapes_for_crop(
        all_plots_in_view + region_entries,
        crop_x, crop_y, crop_w, crop_h, canvas_width, canvas_height,
    )
    n_initial = len(initial_shapes)

    st.caption(f"Zoomed into **{selected_name}** — drawing **{selected_zone}** regions. Click to place points, right-click to finish.")

    canvas_result = st_canvas(
        fill_color=_hex_to_rgba(stroke_color, REGION_ALPHA),
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=crop_img,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        display_toolbar=False,
        initial_drawing={"version": "4.4.0", "objects": initial_shapes},
        key=f"region_canvas_{canvas_width}_{plot['plot_id']}_{selected_zone}",
    )

    if canvas_result.json_data:
        all_paths = [obj for obj in canvas_result.json_data.get("objects", []) if obj.get("type") == "path"]
        new_paths = all_paths[n_initial:]
        if new_paths:
            _render_save_region_form(
                db, garden, plot, new_paths[-1], region_type, selected_zone,
                canvas_width, canvas_height, crop_w, crop_h,
                crop_x, crop_y,
            )

    _render_region_list(db, regions, plot["plot_id"])


# ── Save forms ────────────────────────────────────────────────────────────────

def _render_save_plot_form(
    db: DatabaseConnections,
    garden: dict,
    canvas_obj: dict,
    canvas_width: int,
    canvas_height: int,
    orig_w: int,
    orig_h: int,
) -> None:
    st.divider()
    st.subheader("Save this plot")

    with st.form("save_plot"):
        plot_name = st.text_input("Plot name", placeholder="e.g. Raised Bed 1")
        sun_zone = st.selectbox("Default sun zone", [z.value for z in garden_enum.SunZone])
        height_zone = st.selectbox("Default height zone", [z.value for z in garden_enum.HeightZone])
        save = st.form_submit_button("Save plot", type="primary")

    if save:
        if not plot_name:
            st.error("Plot name is required.")
            return
        pts = _parse_canvas_path(canvas_obj)
        if len(pts) < 3:
            st.error("Draw at least 3 points to define a plot.")
            return
        _save_plot(
            db, garden["garden_id"], plot_name.strip(), pts,
            sun_zone, height_zone,
            canvas_width, canvas_height, orig_w, orig_h,
            garden.get("scale_px_per_foot"),
        )
        st.success(f"Plot '{plot_name}' saved!")
        st.rerun()


def _render_save_region_form(
    db: DatabaseConnections,
    garden: dict,
    plot: dict,
    canvas_obj: dict,
    region_type: str,
    zone_value: str,
    canvas_width: int,
    canvas_height: int,
    shown_w: int,
    shown_h: int,
    crop_x: int = 0,
    crop_y: int = 0,
) -> None:
    st.divider()
    st.subheader(f"Save region: {zone_value}")
    st.caption(f"Plot: {plot['name']}  ·  Type: {region_type}")

    with st.form("save_region"):
        save = st.form_submit_button("Save region", type="primary")

    if save:
        pts = _parse_canvas_path(canvas_obj)
        if len(pts) < 3:
            st.error("Draw at least 3 points.")
            return
        _save_region(
            db, plot["plot_id"], region_type, zone_value, pts,
            canvas_width, canvas_height, shown_w, shown_h,
            garden.get("scale_px_per_foot"),
            crop_x=crop_x, crop_y=crop_y,
        )
        st.success(f"Region '{zone_value}' saved!")
        st.rerun()


# ── Lists ─────────────────────────────────────────────────────────────────────

def _render_plot_list(db: DatabaseConnections, plots: list[dict]) -> None:
    if not plots:
        return
    st.divider()
    st.subheader(f"Plots ({len(plots)})")
    for i, plot in enumerate(plots):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        with st.expander(f"⬛ {plot['name']} — {plot['area_sqft'] or '?'} sqft"):
            st.caption(f"Sun: {plot['sun_zone_default'] or 'not set'}")
            if st.button("🗑️ Delete plot", key=f"del_plot_{plot['plot_id']}", type="secondary"):
                _delete_plot(db, plot["plot_id"])
                st.rerun()


def _render_region_list(db: DatabaseConnections, regions: list[dict], plot_id: str) -> None:
    if not regions:
        st.caption("No regions yet — draw sub-regions above.")
        return
    st.divider()
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.subheader(f"Regions ({len(regions)})")
    with col_clear:
        if st.button("🗑️ Delete all", key=f"del_all_regions_{plot_id}", type="secondary"):
            _delete_all_regions(db, plot_id)
            st.rerun()
    for region in regions:
        icon = "☀️" if region["region_type"] == "sun" else "📐"
        label = f"{icon} {region['zone_value']} — {region['area_sqft'] or '?'} sqft"
        with st.expander(label):
            if st.button("🗑️ Delete region", key=f"del_region_{region['region_id']}", type="secondary"):
                _delete_region(db, region["region_id"])
                st.rerun()
