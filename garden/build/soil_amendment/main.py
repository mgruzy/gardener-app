"""Standalone entry point for the Soil Amendment Advisor.

Run from the garden/ directory:
    cd garden
    python build/soil_amendment/main.py --plot-id <uuid> --plot-name "Bed A"

Or omit --plot-id to list available plots.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import duckdb


# ── Parameters ────────────────────────────────────────────────────────────────

_GARDEN_ROOT    = Path(__file__).parents[2]
_DATA_DIR       = _GARDEN_ROOT / "data"
_GARDEN_DB_PATH = str(_DATA_DIR / "garden.duckdb")

sys.path.insert(0, str(_GARDEN_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_plots() -> list[tuple[str, str]]:
    """Return (plot_id, name) pairs from garden.duckdb."""
    conn = duckdb.connect(_GARDEN_DB_PATH, read_only=True)
    rows = conn.execute("SELECT plot_id, name FROM plots ORDER BY name").fetchall()
    conn.close()
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Soil Amendment Advisor")
    parser.add_argument("--plot-id", help="Plot UUID to analyze")
    parser.add_argument("--plot-name", default="", help="Human-readable plot name")
    args = parser.parse_args()

    if not args.plot_id:
        plots = list_plots()
        if not plots:
            print("No plots found in garden.duckdb.")
            return
        print("Available plots:")
        for pid, name in plots:
            print(f"  {pid}  {name}")
        print("\nRe-run with --plot-id <uuid>")
        return

    plot_name = args.plot_name or args.plot_id

    from build.soil_amendment.amendment_agent import run  # noqa: PLC0415
    advice = asyncio.run(run(args.plot_id, plot_name))
    print("\n" + "=" * 60)
    print(advice)
    print("=" * 60)


if __name__ == "__main__":
    main()
