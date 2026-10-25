"""Per-camera object tracking module for AEGIS-AD."""

from aegis_ad.tracking.models import TrackedFrame, TrackedObject, TrackingError
from aegis_ad.tracking.tracker import ObjectTracker

__all__ = [
    "TrackedFrame",
    "TrackedObject",
    "TrackingError",
    "ObjectTracker",
]
