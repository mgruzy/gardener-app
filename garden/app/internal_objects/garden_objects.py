"""
Garden domain objects.
All dataclasses representing the garden, plots, plants, photos, and snapshots live here.
"""

from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

import app.internal_objects.garden_types as garden_enum


# ---------------------------------------------------------------------------
# Plant library objects (populated by CrewAI plant_db_team agent — Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class PlantType:
    """Top-level plant classification shared across all varieties of the same plant."""

    name: str
    category: garden_enum.PlantCategory
    plant_type_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PlantVariety:
    """
    Agronomic and growing information for a specific plant variety.
    All sow dates are stored as calendar dates after frost date resolution
    by the frost date agent (Phase 2 Step 6).
    """

    plant_type_id: str
    variety_name: str
    plant_category: garden_enum.PlantCategory
    sun_tolerance: garden_enum.SunTolerance
    water_required: garden_enum.WaterRequired
    soil_n: float               # nitrogen level required
    soil_p: float               # phosphorus level required
    soil_k: float               # potassium level required
    growth_needs: str
    post_harvest_soil_needs: str
    days_to_harvest: int
    indoor_sow_weeks_before_frost: int
    outdoor_sow_date_range: str  # e.g. "May 1 - May 15"
    spacing_inches: float
    harvest_timing: str
    temp_min_air_f: float
    temp_min_ground_f: float
    height_inches_estimate: float
    plant_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PlantCompanions:
    """
    Companion and antagonist plant type IDs for a given plant type.
    Both lists contain plant_type_ids referencing entries in the plant_types table.
    Relationships are bidirectional — if Tomato lists Basil as a companion,
    Basil also lists Tomato as a companion.
    """

    plant_type_id: str
    companions: list[str] = field(default_factory=list)   # list of plant_type_ids
    antagonists: list[str] = field(default_factory=list)  # list of plant_type_ids


@dataclass
class PlantPest:
    """A known pest that affects a specific plant type, including symptoms and treatment."""

    plant_type_id: str
    pest_name: str
    symptoms: str
    treatment: str
    pest_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PlantDisease:
    """A known disease that affects a specific plant type, including symptoms and treatment."""

    plant_type_id: str
    disease_name: str
    symptoms: str
    treatment: str
    disease_id: str = field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# Garden state objects (live in garden.duckdb — updated by app)
# ---------------------------------------------------------------------------

@dataclass
class SunZoneRegion:
    """
    A sub-polygon within a plot defining a specific sun exposure area.
    Drawn by the user on the map after the plot polygon is established.
    """

    sun_zone: garden_enum.SunZone
    polygon: list[tuple[float, float]]  # pixel coordinates within the plot


@dataclass
class HeightZoneRegion:
    """
    A sub-polygon within a plot defining a preferred plant height area.
    Drawn by the user on the map after the plot polygon is established.
    """

    height_zone: garden_enum.HeightZone
    polygon: list[tuple[float, float]]  # pixel coordinates within the plot


@dataclass
class Plot:
    """
    A named garden area defined by a drawn polygon on the aerial image.
    Area is computed from the polygon and stored in square feet.
    sun_zone_regions and height_zone_regions are sub-areas drawn within the plot.
    Defaults are used when no sub-regions have been drawn yet.
    """

    garden_id: str
    name: str
    polygon: list[tuple[float, float]]          # pixel coordinates on aerial image
    area_sqft: float                             # computed from polygon at draw time
    sun_zone_default: garden_enum.SunZone
    height_zone_default: garden_enum.HeightZone
    sun_zone_regions: list[SunZoneRegion] = field(default_factory=list)
    height_zone_regions: list[HeightZoneRegion] = field(default_factory=list)
    plant_instances: list["PlantInstance"] = field(default_factory=list)
    notes: str = ""
    plot_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PlantInstance:
    """
    One physical plant — either planned for the ground or already in it.
    status drives the lifecycle: PLANNED → ACTIVE → REMOVED.
    planted_date is None until the plant goes in the ground.
    planned_sow_date is the target date set during planning.
    location_x and location_y are pixel coordinates within the plot polygon.
    """

    plot_id: str
    plant_id: str                       # FK to PlantVariety
    plant_type_id: str                  # denormalized FK to PlantType for easy querying
    location_x: float                   # pixel x within plot polygon
    location_y: float                   # pixel y within plot polygon
    status: garden_enum.PlantStatus = garden_enum.PlantStatus.PLANNED
    planned_sow_date: date | None = None  # target planting date when status is PLANNED
    planted_date: date | None = None      # set when status moves to ACTIVE
    removed_date: date | None = None      # set when status moves to REMOVED
    harvest_count: int = 0
    harvest_weight_lbs: float = 0.0
    instance_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def is_active(self) -> bool:
        """True if the plant is currently in the ground."""
        return self.status == garden_enum.PlantStatus.ACTIVE


@dataclass
class PlantPhoto:
    """
    Metadata record for a photo taken of a plant instance.
    The physical file lives at: data/images/{plant_type}/{name_of_file}
    name_of_file format: {instance_id}_{date_added}.jpg
    """

    plant_instance_id: str
    plant_type_id: str
    plant_id: str
    plant_type: str         # directory name e.g. "tomato" — matches plant_type folder
    name_of_file: str       # e.g. "abc123_2024-05-15.jpg"
    date_added: date
    notes: str = ""
    photo_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class GardenSnapshot:
    """
    Full serialized garden state at a point in time.
    snapshot_json contains all plots and active plant instances.
    Only one snapshot has is_current=True at any time.
    Restore by loading snapshot_json and overwriting garden.duckdb state.
    """

    snapshot_date: date
    triggered_by: garden_enum.SnapshotTrigger
    snapshot_json: str      # full serialized garden state as JSON string
    is_current: bool = False
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class Garden:
    """
    The root object representing the user's property.
    One garden contains multiple plots at different locations on the property.
    lat and lon are derived from zipcode at creation time.
    """

    name: str
    zipcode: str
    aerial_image_path: str
    lat: float
    lon: float
    created_at: date
    plots: list[Plot] = field(default_factory=list)
    garden_id: str = field(default_factory=lambda: str(uuid4()))
