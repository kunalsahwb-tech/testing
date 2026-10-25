"""Data models for per-camera object tracking results."""

from __future__ import annotations

from dataclasses import dataclass


class TrackingError(Exception):
    """Raised when the tracking adapter encounters an unrecoverable error."""


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """A single object with a stable per-camera track ID for the current frame.

    track_id is assigned by ByteTrack and is stable within one camera stream and
    one tracking lifecycle.  It is NOT a cross-camera identity.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.track_id < 1:
            raise ValueError(f"track_id must be a positive integer, got: {self.track_id}")
        if self.class_id < 0:
            raise ValueError(f"class_id cannot be negative, got: {self.class_id}")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got: {self.confidence}"
            )
        if self.x2 <= self.x1:
            raise ValueError(
                f"x2 ({self.x2}) must be greater than x1 ({self.x1})"
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"y2 ({self.y2}) must be greater than y1 ({self.y1})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """Tracking output for a single frame, preserving ingestion and detection metadata."""

    camera_id: str
    frame_index: int
    timestamp_seconds: float
    tracked_objects: list[TrackedObject]

    def __post_init__(self) -> None:
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if self.frame_index < 0:
            raise ValueError(f"frame_index cannot be negative, got: {self.frame_index}")
        if self.timestamp_seconds < 0.0:
            raise ValueError(
                f"timestamp_seconds cannot be negative, got: {self.timestamp_seconds}"
            )
        if not isinstance(self.tracked_objects, list):
            raise ValueError("tracked_objects must be a list.")

    @property
    def tracked_count(self) -> int:
        return len(self.tracked_objects)

    @property
    def track_ids(self) -> frozenset[int]:
        return frozenset(obj.track_id for obj in self.tracked_objects)
