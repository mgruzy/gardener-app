"""Shared canvas utilities for streamlit-drawable-canvas across pages."""

import hashlib
import math

import streamlit.elements.image as _st_image_module
import streamlit.elements.lib.image_utils as _st_image_utils
from PIL import Image, ImageDraw

# ── Compatibility shim (applied once at import time) ──────────────────────────
# streamlit-drawable-canvas passes width as int; Streamlit >=1.37 expects LayoutConfig.
_real_image_to_url = _st_image_utils.image_to_url


def _compat_image_to_url(image, layout_config_or_width, *args, **kwargs):
    if isinstance(layout_config_or_width, int):
        from streamlit.elements.lib.layout_utils import LayoutConfig
        layout_config_or_width = LayoutConfig(width=layout_config_or_width)
    return _real_image_to_url(image, layout_config_or_width, *args, **kwargs)


_st_image_utils.image_to_url = _compat_image_to_url
_st_image_module.image_to_url = _compat_image_to_url


# ── Constants ─────────────────────────────────────────────────────────────────

PLOT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#aaffc3", "#800000", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ff7f50", "#6495ed",
    "#dc143c", "#00ced1", "#9400d3", "#32cd32", "#ff1493",
    "#00bfff", "#ffa500", "#8b0000", "#7fff00", "#db7093",
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

PLANT_STATUS_COLORS = {
    "planned":          "#FFD700",
    "active":           "#3cb44b",
    "fully_harvested":  "#888888",
}


# ── Color helpers ─────────────────────────────────────────────────────────────

def plant_type_color(plant_type_id: str) -> str:
    """Pick a deterministic color from PLOT_COLORS for a given plant_type_id."""
    raw = hashlib.md5(plant_type_id.encode()).digest()
    idx = int.from_bytes(raw[:4], "big") % len(PLOT_COLORS)
    return PLOT_COLORS[idx]


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Image crop ────────────────────────────────────────────────────────────────

def get_plot_crop(
    img: Image.Image,
    polygon_orig: list,
    canvas_width: int,
    padding_frac: float = 0.25,
) -> tuple[Image.Image, int, int, int, int]:
    """Crop the original image to the plot bounding box with padding.

    Returns (cropped_img_resized_to_canvas_width, crop_x1, crop_y1, crop_w, crop_h).
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


# ── Coordinate helpers ────────────────────────────────────────────────────────

def orig_to_canvas(
    orig_x: float, orig_y: float,
    crop_x: int, crop_y: int, crop_w: int, crop_h: int,
    canvas_w: int, canvas_h: int,
) -> tuple[float, float]:
    """Convert a point in original image space to cropped canvas space."""
    return (
        (orig_x - crop_x) / crop_w * canvas_w,
        (orig_y - crop_y) / crop_h * canvas_h,
    )


def canvas_to_orig(
    cx: float, cy: float,
    crop_x: int, crop_y: int, crop_w: int, crop_h: int,
    canvas_w: int, canvas_h: int,
) -> tuple[float, float]:
    """Convert a point in cropped canvas space to original image space."""
    return (
        cx / canvas_w * crop_w + crop_x,
        cy / canvas_h * crop_h + crop_y,
    )


def spacing_to_canvas_px(
    spacing_inches: float,
    px_per_foot: float | None,
    crop_w: int,
    canvas_w: int,
) -> float:
    """Convert plant spacing (inches) to canvas pixels in the cropped view."""
    if px_per_foot and px_per_foot > 0:
        px_per_inch_orig = px_per_foot / 12.0
        px_per_inch_canvas = px_per_inch_orig * canvas_w / crop_w
        return spacing_inches * px_per_inch_canvas
    return max(6.0, spacing_inches * 1.5)  # rough fallback


# ── Canvas shape builders ─────────────────────────────────────────────────────

def shapes_for_crop(
    polygons_orig: list[tuple[list, str, float]],
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    canvas_w: int,
    canvas_h: int,
) -> list[dict]:
    """Convert (pts, stroke_color, fill_alpha) polygon tuples to canvas path objects."""
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
            "stroke": color, "fill": hex_to_rgba(color, alpha),
            "strokeWidth": 2, "selectable": False,
        })
    return shapes


def plant_circle_shape(
    canvas_x: float,
    canvas_y: float,
    radius_px: float,
    color: str,
    status: str = "active",
) -> dict:
    """Build a fabric.js circle dict for a plant instance marker.

    active           → solid fill
    planned          → transparent fill (25% opacity)
    fully_harvested  → outline only (no fill)
    """
    r = max(4.0, radius_px)
    if status == "active":
        fill = hex_to_rgba(color, 0.90)
        stroke = "#ffffff"
        stroke_width = 1.5
    elif status == "planned":
        fill = hex_to_rgba(color, 0.25)
        stroke = color
        stroke_width = 2.0
    else:  # fully_harvested
        fill = "rgba(0,0,0,0)"
        stroke = color
        stroke_width = 2.5
    return {
        "type": "circle",
        "left": canvas_x - r,
        "top": canvas_y - r,
        "radius": r,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width,
        "selectable": False,
    }


def row_line_shape(
    x1: float, y1: float, x2: float, y2: float,
    color: str,
) -> dict:
    """Build a fabric.js line dict connecting row plant markers."""
    return {
        "type": "line",
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "left": min(x1, x2), "top": min(y1, y2),
        "stroke": color, "strokeWidth": 1,
        "strokeDashArray": [4, 3],
        "selectable": False,
    }


# ── Row geometry ──────────────────────────────────────────────────────────────

def row_plant_positions(
    x1: float, y1: float, x2: float, y2: float,
    spacing_px: float,
) -> list[tuple[float, float]]:
    """Return canvas (x, y) positions for plants along a row at spacing_px intervals."""
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1 or spacing_px < 1:
        return [(x1, y1)]
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    n = max(1, int(length / spacing_px) + 1)
    return [(x1 + i * spacing_px * dx, y1 + i * spacing_px * dy) for i in range(n)]


# ── Path parsing ──────────────────────────────────────────────────────────────

def parse_canvas_path(obj: dict) -> list[list[float]]:
    """Extract [x, y] points from a drawable-canvas polygon path object."""
    path = obj.get("path", [])
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


# ── PIL composite ─────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def composite_plant_canvas(
    base_img: Image.Image,
    poly_shapes_canvas: list[tuple[list, str, float]],
    instance_data: list[dict],
    row_line_data: list[tuple[float, float, float, float, str]],
) -> Image.Image:
    """Render region overlays and plant circles directly onto base_img.

    Eliminates the need for fabric.js initial_drawing objects, which fail to
    initialize when the canvas is inside a hidden Streamlit tab.

    Args:
        base_img: Cropped background image (already in canvas pixel space).
        poly_shapes_canvas: List of (pts_in_canvas_space, hex_color, alpha) for plot/regions.
        instance_data: Per-plant dicts with keys cx, cy, radius, color, status, selected.
        row_line_data: Dashed connectors as (x1, y1, x2, y2, hex_color).
    """
    w, h = base_img.width, base_img.height
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for pts, color, alpha in poly_shapes_canvas:
        if not pts:
            continue
        flat = [(float(p[0]), float(p[1])) for p in pts]
        r, g, b = _hex_to_rgb(color)
        draw.polygon(flat, fill=(r, g, b, int(alpha * 255)), outline=(r, g, b, 200))

    for x1, y1, x2, y2, color in row_line_data:
        r, g, b = _hex_to_rgb(color)
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 1:
            continue
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        dash, gap, t, on = 4, 3, 0.0, True
        while t < length:
            seg = min(t + (dash if on else gap), length)
            if on:
                draw.line(
                    [(x1 + t * dx, y1 + t * dy), (x1 + seg * dx, y1 + seg * dy)],
                    fill=(r, g, b, 180), width=1,
                )
            t, on = seg, not on

    for inst in instance_data:
        cx, cy, radius = inst["cx"], inst["cy"], max(4.0, inst["radius"])
        r, g, b = _hex_to_rgb(inst["color"])
        status = inst["status"]
        is_sel = inst.get("selected", False)

        if status == "active":
            fill = (r, g, b, int(0.90 * 255))
            stroke = (255, 255, 255, 255)
            stroke_w = 3 if is_sel else 1
        elif status == "planned":
            fill = (r, g, b, int(0.25 * 255))
            stroke = (255, 255, 255, 255) if is_sel else (r, g, b, 255)
            stroke_w = 3 if is_sel else 2
        else:  # fully_harvested
            fill = (0, 0, 0, 0)
            stroke = (255, 255, 255, 255) if is_sel else (r, g, b, 255)
            stroke_w = 3 if is_sel else 2

        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=fill, outline=stroke, width=stroke_w)

    result = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    return result.convert("RGB")
