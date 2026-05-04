"""
Weather domain objects.
Dataclasses for weather observations and forecast records stored in weather.duckdb.
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class WeatherRecord:
    """
    Daily weather observation or forecast for a zipcode.
    Imperial units throughout: °F, inches, mph, inHg.
    """

    zipcode: str
    date: date
    temp_min_f: float
    temp_max_f: float
    precipitation_in: float
    humidity_pct: float
    pressure_inhg: float
    wind_min_mph: float
    wind_max_mph: float
    wind_direction: str     # e.g. "NW"
    air_quality_index: float


@dataclass
class WeatherMetadata:
    """
    Tracks the date range of weather data loaded for a zipcode.
    Used by the weather agent to determine what dates still need to be fetched.
    """

    zipcode: str
    first_record_date: date
    last_updated_date: date
