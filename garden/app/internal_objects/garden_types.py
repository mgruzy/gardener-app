"""
Enums for all categorical choices across the garden app models.
All free-text categoricals are represented here — never raw strings in model fields.
"""

from enum import Enum


class PlantCategory(str, Enum):
    HERB = "herb"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"


class SunTolerance(str, Enum):
    FULL_SUN = "full_sun"
    PARTIAL_SHADE = "partial_shade"
    FULL_SHADE = "full_shade"


class WaterRequired(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SunZone(str, Enum):
    FULL_SUN = "full_sun"
    PARTIAL_SHADE = "partial_shade"
    FULL_SHADE = "full_shade"


class HeightZone(str, Enum):
    TALL = "tall"
    MEDIUM = "medium"
    LOW = "low"


class SnapshotTrigger(str, Enum):
    MANUAL = "manual"
    AUTO_DAILY = "auto_daily"
    PRE_SEASON = "pre_season"
    PRE_RESTORE = "pre_restore"


class PlantStatus(str, Enum):
    PLANNED = "planned"   # intended to plant — has a planned_sow_date, not yet in ground
    ACTIVE = "active"     # in the ground — has a planted_date
    REMOVED = "removed"   # pulled out — has a removed_date


class AppMode(str, Enum):
    LIVE = "live"
    TEST = "test"
