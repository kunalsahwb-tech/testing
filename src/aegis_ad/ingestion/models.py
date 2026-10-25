"""Data models for video ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


class IngestionError(Exception):
    """Raised when video ingestion or decoding encounters an error."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata describing a video source."""

    source_path: str
    camera_id: str
    width: int
    height: int
    fps: float
    total_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: width={self.width}, height={self.height}")
        if self.fps <= 0.0:
            raise ValueError(f"FPS must be positive, got: {self.fps}")
        if self.total_frames is not None and self.total_frames < 0:
            raise ValueError(f"total_frames cannot be negative, got: {self.total_frames}")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Represents a single frame extracted from a video source."""

    camera_id: str
    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if self.frame_index < 0:
            raise ValueError(f"frame_index cannot be negative, got: {self.frame_index}")
        if self.timestamp_seconds < 0.0:
            raise ValueError(f"timestamp_seconds cannot be negative, got: {self.timestamp_seconds}")
        if not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
            raise ValueError("frame must be a non-empty numpy.ndarray.")
        if not self.source:
            raise ValueError("source reference cannot be empty.")
