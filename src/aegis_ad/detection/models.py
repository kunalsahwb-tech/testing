"""Data models for object detection results."""

from __future__ import annotations

from dataclasses import dataclass, field


class DetectionError(Exception):
    """Raised when model loading or inference fails."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name, val in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2)):
            if not isinstance(val, (int, float)) or val != val:  # NaN check
                raise ValueError(f"BoundingBox.{name} must be a finite number, got: {val!r}")
        if self.x2 <= self.x1:
            raise ValueError(
                f"BoundingBox x2 ({self.x2}) must be greater than x1 ({self.x1})"
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"BoundingBox y2 ({self.y2}) must be greater than y1 ({self.y1})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object within a frame."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError(f"class_id cannot be negative, got: {self.class_id}")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got: {self.confidence}"
            )
        if not isinstance(self.bounding_box, BoundingBox):
            raise ValueError("bounding_box must be a BoundingBox instance.")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detection output for a single frame, preserving ingestion metadata."""

    camera_id: str
    frame_index: int
    timestamp_seconds: float
    detections: list[Detection]
    inference_time_ms: float

    def __post_init__(self) -> None:
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if self.frame_index < 0:
            raise ValueError(f"frame_index cannot be negative, got: {self.frame_index}")
        if self.timestamp_seconds < 0.0:
            raise ValueError(
                f"timestamp_seconds cannot be negative, got: {self.timestamp_seconds}"
            )
        if not isinstance(self.detections, list):
            raise ValueError("detections must be a list.")
        if self.inference_time_ms < 0.0:
            raise ValueError(
                f"inference_time_ms cannot be negative, got: {self.inference_time_ms}"
            )

    @property
    def detection_count(self) -> int:
        return len(self.detections)
