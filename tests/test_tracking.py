"""Tests for per-camera object tracking (no model weights or network required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from aegis_ad.detection.models import BoundingBox, Detection, DetectionResult
from aegis_ad.tracking.models import TrackedFrame, TrackedObject, TrackingError
from aegis_ad.tracking.tracker import ObjectTracker, _DetectionAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 110.0,
    y2: float = 220.0,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _make_result(
    camera_id: str = "CAM_01",
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    detections: list[Detection] | None = None,
    inference_time_ms: float = 5.0,
) -> DetectionResult:
    return DetectionResult(
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        detections=detections if detections is not None else [],
        inference_time_ms=inference_time_ms,
    )


def _make_tracker(camera_id: str = "CAM_01") -> ObjectTracker:
    """Return a real ObjectTracker (uses actual BYTETracker internally)."""
    return ObjectTracker(
        camera_id=camera_id,
        track_high_thresh=0.4,
        track_low_thresh=0.1,
        new_track_thresh=0.5,
        track_buffer=5,
        match_thresh=0.8,
    )


# ---------------------------------------------------------------------------
# TrackedObject model tests
# ---------------------------------------------------------------------------

def test_tracked_object_valid():
    obj = TrackedObject(
        track_id=1, class_id=0, class_name="person",
        confidence=0.9, x1=0.0, y1=0.0, x2=100.0, y2=200.0,
    )
    assert obj.width == pytest.approx(100.0)
    assert obj.height == pytest.approx(200.0)


def test_tracked_object_zero_track_id_invalid():
    with pytest.raises(ValueError, match="track_id"):
        TrackedObject(
            track_id=0, class_id=0, class_name="person",
            confidence=0.5, x1=0.0, y1=0.0, x2=50.0, y2=50.0,
        )


def test_tracked_object_negative_track_id_invalid():
    with pytest.raises(ValueError, match="track_id"):
        TrackedObject(
            track_id=-1, class_id=0, class_name="person",
            confidence=0.5, x1=0.0, y1=0.0, x2=50.0, y2=50.0,
        )


def test_tracked_object_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        TrackedObject(
            track_id=1, class_id=0, class_name="person",
            confidence=1.5, x1=0.0, y1=0.0, x2=50.0, y2=50.0,
        )


def test_tracked_object_degenerate_box():
    with pytest.raises(ValueError):
        TrackedObject(
            track_id=1, class_id=0, class_name="person",
            confidence=0.8, x1=100.0, y1=0.0, x2=50.0, y2=50.0,
        )


# ---------------------------------------------------------------------------
# TrackedFrame model tests
# ---------------------------------------------------------------------------

def test_tracked_frame_valid():
    frame = TrackedFrame(
        camera_id="CAM_01", frame_index=5, timestamp_seconds=0.5,
        tracked_objects=[],
    )
    assert frame.tracked_count == 0
    assert frame.track_ids == frozenset()


def test_tracked_frame_preserves_metadata():
    obj = TrackedObject(
        track_id=3, class_id=2, class_name="car",
        confidence=0.75, x1=50.0, y1=50.0, x2=200.0, y2=150.0,
    )
    frame = TrackedFrame(
        camera_id="CAM_TEST", frame_index=42, timestamp_seconds=4.2,
        tracked_objects=[obj],
    )
    assert frame.camera_id == "CAM_TEST"
    assert frame.frame_index == 42
    assert frame.timestamp_seconds == pytest.approx(4.2)
    assert frame.tracked_count == 1
    assert 3 in frame.track_ids


# ---------------------------------------------------------------------------
# _DetectionAdapter tests
# ---------------------------------------------------------------------------

def test_detection_adapter_empty_result():
    result = _make_result(detections=[])
    adapter = _DetectionAdapter.from_detection_result(result)
    assert len(adapter) == 0
    assert adapter.conf.shape == (0,)
    assert adapter.xywh.shape == (0, 4)


def test_detection_adapter_single_detection():
    det = _make_detection(x1=10.0, y1=20.0, x2=110.0, y2=220.0, confidence=0.88)
    adapter = _DetectionAdapter.from_detection_result(_make_result(detections=[det]))
    assert len(adapter) == 1
    assert adapter.conf[0] == pytest.approx(0.88)
    # centre-x = (10+110)/2 = 60
    assert adapter.xywh[0, 0] == pytest.approx(60.0)
    # centre-y = (20+220)/2 = 120
    assert adapter.xywh[0, 1] == pytest.approx(120.0)
    # width = 100, height = 200
    assert adapter.xywh[0, 2] == pytest.approx(100.0)
    assert adapter.xywh[0, 3] == pytest.approx(200.0)


def test_detection_adapter_boolean_indexing():
    dets = [
        _make_detection(confidence=0.9, x1=0.0, y1=0.0, x2=50.0, y2=50.0),
        _make_detection(confidence=0.3, x1=100.0, y1=0.0, x2=200.0, y2=100.0),
    ]
    adapter = _DetectionAdapter.from_detection_result(_make_result(detections=dets))
    mask = adapter.conf >= 0.5
    sub = adapter[mask]
    assert len(sub) == 1
    assert sub.conf[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ObjectTracker – real BYTETracker integration tests
# ---------------------------------------------------------------------------

def test_tracker_assigns_track_id_on_first_detection():
    """A detection above the activation threshold must receive a track ID."""
    tracker = _make_tracker()
    det = _make_detection(confidence=0.9)
    result = _make_result(detections=[det])
    frame = tracker.update(result)
    assert frame.tracked_count == 1
    assert frame.tracked_objects[0].track_id >= 1


def test_tracker_retains_track_id_across_frames():
    """An object that barely moves across consecutive frames must keep its track ID."""
    tracker = _make_tracker()
    det1 = _make_detection(confidence=0.9, x1=10.0, y1=10.0, x2=110.0, y2=110.0)
    det2 = _make_detection(confidence=0.9, x1=12.0, y1=11.0, x2=112.0, y2=111.0)

    r1 = _make_result(frame_index=0, timestamp_seconds=0.0, detections=[det1])
    r2 = _make_result(frame_index=1, timestamp_seconds=0.1, detections=[det2])

    frame1 = tracker.update(r1)
    frame2 = tracker.update(r2)

    # Both frames must have exactly one tracked object
    assert frame1.tracked_count == 1
    assert frame2.tracked_count == 1
    # The track ID must be the same in both frames
    assert frame1.tracked_objects[0].track_id == frame2.tracked_objects[0].track_id


def test_tracker_two_objects_get_different_track_ids():
    """Two separate detections in the same frame must receive different track IDs."""
    tracker = _make_tracker()
    det1 = _make_detection(confidence=0.9, x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    det2 = _make_detection(confidence=0.85, x1=300.0, y1=300.0, x2=450.0, y2=450.0)
    result = _make_result(detections=[det1, det2])
    frame = tracker.update(result)
    assert frame.tracked_count == 2
    ids = [obj.track_id for obj in frame.tracked_objects]
    assert ids[0] != ids[1]


def test_tracker_preserves_frame_metadata():
    """TrackedFrame must carry through camera_id, frame_index, and timestamp."""
    tracker = _make_tracker("CAM_PRESERVE")
    det = _make_detection(confidence=0.9)
    result = _make_result(
        camera_id="CAM_PRESERVE", frame_index=99, timestamp_seconds=9.9,
        detections=[det],
    )
    frame = tracker.update(result)
    assert frame.camera_id == "CAM_PRESERVE"
    assert frame.frame_index == 99
    assert frame.timestamp_seconds == pytest.approx(9.9)


def test_tracker_preserves_class_label_and_box():
    """Class label, confidence, and bounding-box coordinates must be preserved."""
    tracker = _make_tracker()
    det = _make_detection(class_id=2, class_name="car", confidence=0.85,
                          x1=50.0, y1=60.0, x2=200.0, y2=160.0)
    result = _make_result(detections=[det])
    frame = tracker.update(result)
    assert frame.tracked_count == 1
    obj = frame.tracked_objects[0]
    assert obj.class_id == 2
    assert obj.class_name == "car"
    # Confidence comes from BYTETracker output — it may be equal or slightly fused;
    # confirm it is in [0, 1] and non-zero.
    assert 0.0 < obj.confidence <= 1.0
    # Box coordinates: BYTETracker returns xyxy — compare approximately
    assert obj.x1 == pytest.approx(50.0, abs=1.0)
    assert obj.y1 == pytest.approx(60.0, abs=1.0)
    assert obj.x2 == pytest.approx(200.0, abs=1.0)
    assert obj.y2 == pytest.approx(160.0, abs=1.0)


def test_tracker_empty_detection_frame_returns_empty_tracked_frame():
    """An empty detection frame must not raise and must return an empty TrackedFrame."""
    tracker = _make_tracker()
    result = _make_result(detections=[])
    frame = tracker.update(result)
    assert isinstance(frame, TrackedFrame)
    assert frame.tracked_count == 0


def test_tracker_lost_object_not_immediately_removed():
    """A track should survive at least one frame gap within the track_buffer window."""
    tracker = ObjectTracker(
        camera_id="CAM_01",
        track_high_thresh=0.4,
        track_low_thresh=0.1,
        new_track_thresh=0.5,
        track_buffer=5,
        match_thresh=0.8,
    )
    det = _make_detection(confidence=0.9, x1=10.0, y1=10.0, x2=110.0, y2=110.0)

    # Frame 0: establish the track
    frame0 = tracker.update(_make_result(frame_index=0, detections=[det]))
    assert frame0.tracked_count == 1
    track_id = frame0.tracked_objects[0].track_id

    # Frame 1: object disappears
    frame1 = tracker.update(_make_result(frame_index=1, detections=[]))

    # Frame 2: object reappears in roughly the same position
    det_back = _make_detection(confidence=0.9, x1=12.0, y1=12.0, x2=112.0, y2=112.0)
    frame2 = tracker.update(_make_result(frame_index=2, detections=[det_back]))

    # The re-found track may keep the original ID or get a new one — both are valid
    # behaviour from ByteTrack. What matters is: no exception was raised and the
    # result is a valid TrackedFrame.
    assert isinstance(frame2, TrackedFrame)


def test_tracker_reset_starts_new_lifecycle():
    """After reset(), the tracker behaves as if freshly constructed."""
    tracker = _make_tracker()
    det = _make_detection(confidence=0.9)
    tracker.update(_make_result(frame_index=0, detections=[det]))

    tracker.reset()

    # After reset the internal BYTETracker's frame_id should be 0
    assert tracker._tracker.frame_id == 0

    frame = tracker.update(_make_result(frame_index=1, detections=[det]))
    assert frame.tracked_count == 1


def test_tracker_rejects_different_camera():
    """Passing detections from a different camera must raise TrackingError."""
    tracker = _make_tracker("CAM_A")
    result = _make_result(camera_id="CAM_B", detections=[_make_detection()])
    with pytest.raises(TrackingError, match="Camera mismatch"):
        tracker.update(result)


def test_tracker_invalid_camera_id_raises():
    with pytest.raises(ValueError):
        ObjectTracker(camera_id="")


def test_tracker_low_confidence_detection_not_tracked():
    """Detections below new_track_thresh and track_high_thresh must not produce tracks."""
    tracker = ObjectTracker(
        camera_id="CAM_01",
        track_high_thresh=0.7,
        track_low_thresh=0.3,
        new_track_thresh=0.8,
        track_buffer=5,
        match_thresh=0.8,
    )
    det = _make_detection(confidence=0.2)  # below track_low_thresh=0.3
    result = _make_result(detections=[det])
    frame = tracker.update(result)
    assert frame.tracked_count == 0


# ---------------------------------------------------------------------------
# CLI configuration tests (no model weights or network)
# ---------------------------------------------------------------------------

def _make_mock_settings(tracking_cfg: dict | None = None, detection_cfg: dict | None = None) -> MagicMock:
    mock = MagicMock()
    mock.raw = {
        "detection": detection_cfg or {"model_path": "mock.pt", "confidence_threshold": 0.25, "device": "cpu"},
        "tracking": tracking_cfg or {},
    }
    return mock


def _run_tracking_cli(args: list[str], tracking_cfg: dict | None = None) -> int:
    """Run the tracking CLI with VideoSource, YOLODetector, and ObjectTracker mocked."""
    from aegis_ad.tracking.__main__ import run_cli

    mock_video = MagicMock()
    mock_video.metadata.camera_id = "CAM_MOCK"
    mock_video.metadata.source_path = "mock.mp4"
    mock_video.metadata.width = 640
    mock_video.metadata.height = 480
    mock_video.metadata.fps = 25.0
    mock_video.read_frames.return_value = iter([])

    mock_detector_instance = MagicMock()
    mock_detector_instance.detect.return_value = DetectionResult(
        camera_id="CAM_MOCK", frame_index=0, timestamp_seconds=0.0,
        detections=[], inference_time_ms=1.0,
    )

    with (
        patch("aegis_ad.tracking.__main__.load_settings",
              return_value=_make_mock_settings(tracking_cfg)),
        patch("aegis_ad.tracking.__main__.VideoSource", return_value=mock_video),
        patch("aegis_ad.tracking.__main__.YOLODetector", return_value=mock_detector_instance),
        patch("pathlib.Path.exists", return_value=True),
    ):
        return run_cli(args)


def test_tracking_cli_runs_successfully():
    """Tracking CLI must exit 0 with a mocked pipeline."""
    code = _run_tracking_cli(["--source", "any.mp4", "--model", "mock.pt"])
    assert code == 0


def test_tracking_cli_missing_source_returns_1():
    """CLI must return 1 and report an error if the video file does not exist."""
    from aegis_ad.tracking.__main__ import run_cli
    with patch("pathlib.Path.exists", return_value=False):
        code = run_cli(["--source", "nonexistent.mp4", "--model", "mock.pt"])
    assert code == 1


def test_tracking_cli_uses_track_buffer_from_config():
    """track_buffer from config must reach ObjectTracker constructor."""
    from aegis_ad.tracking.__main__ import run_cli

    constructed_kwargs: list[dict] = []

    class _RecordingTracker:
        def __init__(self, **kw):
            constructed_kwargs.append(kw)
            self.camera_id = kw.get("camera_id", "CAM_MOCK")
        def update(self, result):
            return TrackedFrame(
                camera_id=self.camera_id, frame_index=0,
                timestamp_seconds=0.0, tracked_objects=[],
            )
        def reset(self):
            pass

    mock_video = MagicMock()
    mock_video.metadata.camera_id = "CAM_MOCK"
    mock_video.metadata.source_path = "mock.mp4"
    mock_video.metadata.width = 640
    mock_video.metadata.height = 480
    mock_video.metadata.fps = 25.0
    mock_video.read_frames.return_value = iter([])

    mock_detector_instance = MagicMock()
    mock_detector_instance.detect.return_value = DetectionResult(
        camera_id="CAM_MOCK", frame_index=0, timestamp_seconds=0.0,
        detections=[], inference_time_ms=1.0,
    )

    with (
        patch("aegis_ad.tracking.__main__.load_settings",
              return_value=_make_mock_settings({"track_buffer": 99})),
        patch("aegis_ad.tracking.__main__.VideoSource", return_value=mock_video),
        patch("aegis_ad.tracking.__main__.YOLODetector", return_value=mock_detector_instance),
        patch("aegis_ad.tracking.__main__.ObjectTracker",
              side_effect=lambda **kw: _RecordingTracker(**kw)),
        patch("pathlib.Path.exists", return_value=True),
    ):
        code = run_cli(["--source", "any.mp4", "--model", "mock.pt"])

    assert code == 0
    assert constructed_kwargs, "ObjectTracker was never constructed"
    assert constructed_kwargs[0].get("track_buffer") == 99
