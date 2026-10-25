"""Per-camera ByteTrack adapter for AEGIS-AD."""

from __future__ import annotations

import logging
import types
from typing import Any

import numpy as np

from aegis_ad.detection.models import DetectionResult
from aegis_ad.detection.detector import COCO_CLASS_NAMES
from aegis_ad.tracking.models import TrackedFrame, TrackedObject, TrackingError

logger = logging.getLogger(__name__)

# BYTETracker output column indices
_COL_X1 = 0
_COL_Y1 = 1
_COL_X2 = 2
_COL_Y2 = 3
_COL_TRACK_ID = 4
_COL_SCORE = 5
_COL_CLS = 6
_COL_IDX = 7


class _DetectionAdapter:
    """Lightweight wrapper that presents a DetectionResult in the format BYTETracker.update expects.

    BYTETracker requires an object with:
      - .conf   : np.ndarray  shape (N,)  float32 confidence scores
      - .cls    : np.ndarray  shape (N,)  float32 class IDs
      - .xywh   : np.ndarray  shape (N,4) float32 centre-x, centre-y, width, height
      - boolean indexing via __getitem__
    """

    def __init__(
        self,
        conf: np.ndarray,
        cls: np.ndarray,
        xywh: np.ndarray,
    ) -> None:
        self.conf = conf
        self.cls = cls
        self.xywh = xywh

    def __getitem__(self, mask: Any) -> "_DetectionAdapter":
        return _DetectionAdapter(
            conf=self.conf[mask],
            cls=self.cls[mask],
            xywh=self.xywh[mask],
        )

    def __len__(self) -> int:
        return len(self.conf)

    @classmethod
    def from_detection_result(cls, result: DetectionResult) -> "_DetectionAdapter":
        """Convert a DetectionResult into the adapter format."""
        if not result.detections:
            empty = np.empty((0,), dtype=np.float32)
            return cls(
                conf=empty,
                cls=empty,
                xywh=np.empty((0, 4), dtype=np.float32),
            )

        conf_list: list[float] = []
        cls_list: list[float] = []
        xywh_rows: list[list[float]] = []

        for det in result.detections:
            bb = det.bounding_box
            cx = (bb.x1 + bb.x2) / 2.0
            cy = (bb.y1 + bb.y2) / 2.0
            w = bb.x2 - bb.x1
            h = bb.y2 - bb.y1
            conf_list.append(det.confidence)
            cls_list.append(float(det.class_id))
            xywh_rows.append([cx, cy, w, h])

        return cls(
            conf=np.array(conf_list, dtype=np.float32),
            cls=np.array(cls_list, dtype=np.float32),
            xywh=np.array(xywh_rows, dtype=np.float32),
        )


def _make_bytetrack_args(
    track_high_thresh: float = 0.5,
    track_low_thresh: float = 0.1,
    new_track_thresh: float = 0.6,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    fuse_score: bool = True,
) -> types.SimpleNamespace:
    """Return a SimpleNamespace that satisfies BYTETracker.__init__ requirements."""
    return types.SimpleNamespace(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
    )


class ObjectTracker:
    """Per-camera ByteTrack adapter.

    One ObjectTracker instance is bound to a single camera stream identified by
    camera_id.  Passing detections from a different camera raises TrackingError
    to prevent silent cross-camera state corruption.

    track_id values are assigned by ByteTrack and are stable within one camera
    stream and one tracking lifecycle.  They are NOT cross-camera identities.
    """

    def __init__(
        self,
        camera_id: str,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        """Initialise the tracker for a single camera stream.

        Args:
            camera_id: Camera this tracker is bound to.
            track_high_thresh: High-confidence detection threshold for track activation.
            track_low_thresh: Low-confidence threshold; detections below this are ignored.
            new_track_thresh: Minimum confidence to start a new track.
            track_buffer: Number of frames to keep a lost track before removing it.
            match_thresh: IoU threshold for matching detections to existing tracks.
        """
        if not camera_id or not isinstance(camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")

        self.camera_id = camera_id
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh = track_low_thresh
        self._new_track_thresh = new_track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh

        self._tracker = self._build_tracker()

    def _build_tracker(self):  # type: ignore[return]
        """Construct a fresh BYTETracker with the configured parameters."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: PLC0415
        except Exception as exc:
            raise TrackingError(f"Failed to import BYTETracker: {exc}") from exc

        args = _make_bytetrack_args(
            track_high_thresh=self._track_high_thresh,
            track_low_thresh=self._track_low_thresh,
            new_track_thresh=self._new_track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
        )
        return BYTETracker(args)

    def reset(self) -> None:
        """Reset the tracker, starting a new tracking lifecycle.

        All existing track IDs are discarded. Use between different videos or streams.
        """
        self._tracker = self._build_tracker()
        logger.debug("[%s] Tracker reset.", self.camera_id)

    def update(self, result: DetectionResult) -> TrackedFrame:
        """Update the tracker with detections from one frame and return tracked objects.

        Args:
            result: Detection output for the current frame.

        Returns:
            TrackedFrame with track IDs and preserved metadata.

        Raises:
            TrackingError: If detections are from a different camera, or on tracker failure.
        """
        if result.camera_id != self.camera_id:
            raise TrackingError(
                f"Camera mismatch: tracker is bound to '{self.camera_id}' "
                f"but received detections from '{result.camera_id}'."
            )

        adapter = _DetectionAdapter.from_detection_result(result)

        try:
            raw_output = self._tracker.update(adapter)
        except Exception as exc:
            raise TrackingError(
                f"[{self.camera_id}] Frame {result.frame_index}: tracker update failed: {exc}"
            ) from exc

        tracked_objects = self._parse_output(raw_output, result)

        logger.debug(
            "[%s] Frame %d: %d detection(s) → %d track(s)",
            self.camera_id,
            result.frame_index,
            result.detection_count,
            len(tracked_objects),
        )

        return TrackedFrame(
            camera_id=result.camera_id,
            frame_index=result.frame_index,
            timestamp_seconds=result.timestamp_seconds,
            tracked_objects=tracked_objects,
        )

    def _parse_output(
        self,
        raw: np.ndarray,
        result: DetectionResult,
    ) -> list[TrackedObject]:
        """Convert BYTETracker output array to TrackedObject instances.

        Columns: [x1, y1, x2, y2, track_id, score, cls, idx]
        """
        tracked: list[TrackedObject] = []

        if raw is None or len(raw) == 0:
            return tracked

        for row in raw:
            track_id = int(row[_COL_TRACK_ID])
            if track_id < 1:
                logger.debug("Skipping row with unassigned track_id=%d", track_id)
                continue

            score = float(row[_COL_SCORE])
            class_id = int(row[_COL_CLS])
            x1, y1, x2, y2 = (
                float(row[_COL_X1]),
                float(row[_COL_Y1]),
                float(row[_COL_X2]),
                float(row[_COL_Y2]),
            )

            # Guard degenerate boxes that ByteTrack might occasionally extrapolate
            if x2 <= x1 or y2 <= y1:
                logger.debug(
                    "Skipping degenerate tracked box: track_id=%d [%.1f,%.1f,%.1f,%.1f]",
                    track_id, x1, y1, x2, y2,
                )
                continue

            class_name = COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")

            try:
                tracked.append(
                    TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=max(0.0, min(1.0, score)),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
            except ValueError as exc:
                logger.warning("Skipping invalid tracked object (track_id=%d): %s", track_id, exc)

        return tracked
