"""Event detection module for AEGIS-AD."""

from aegis_ad.events.engine import EventEngine
from aegis_ad.events.models import (
    EventConfigurationError,
    RestrictedZone,
    SafetyEvent,
    VALID_EVENT_TYPES,
    point_in_polygon,
)

__all__ = [
    "EventConfigurationError",
    "EventEngine",
    "RestrictedZone",
    "SafetyEvent",
    "VALID_EVENT_TYPES",
    "point_in_polygon",
]
