"""Tests for detection data models (no model weights required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from aegis_ad.detection.models import (
    BoundingBox,
    Detection,
    DetectionError,
    DetectionResult,
)
from aegis_ad.detection.detector import COCO_CLASS_NAMES, TARGET_CLASS_IDS, YOLODetector
from aegis_ad.ingestion.models import FramePacket


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------

def test_bounding_box_valid():
    bb = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
    assert bb.width == pytest.approx(90.0)
    assert bb.height == pytest.approx(180.0)
    assert bb.area == pytest.approx(16200.0)


def test_bounding_box_x2_must_exceed_x1():
    with pytest.raises(ValueError, match="x2.*greater than x1"):
        BoundingBox(x1=100.0, y1=0.0, x2=50.0, y2=100.0)


def test_bounding_box_y2_must_exceed_y1():
    with pytest.raises(ValueError, match="y2.*greater than y1"):
        BoundingBox(x1=0.0, y1=100.0, x2=100.0, y2=50.0)


def test_bounding_box_equal_coords_invalid():
    with pytest.raises(ValueError):
        BoundingBox(x1=10.0, y1=10.0, x2=10.0, y2=50.0)


def test_bounding_box_nan_rejected():
    with pytest.raises(ValueError, match="finite number"):
        BoundingBox(x1=float("nan"), y1=0.0, x2=100.0, y2=100.0)


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

def test_detection_valid():
    bb = BoundingBox(0.0, 0.0, 50.0, 80.0)
    det = Detection(class_id=0, class_name="person", confidence=0.85, bounding_box=bb)
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.85)


def test_detection_negative_class_id_invalid():
    bb = BoundingBox(0.0, 0.0, 50.0, 80.0)
    with pytest.raises(ValueError, match="class_id"):
        Detection(class_id=-1, class_name="person", confidence=0.5, bounding_box=bb)


def test_detection_confidence_above_one_invalid():
    bb = BoundingBox(0.0, 0.0, 50.0, 80.0)
    with pytest.raises(ValueError, match="confidence"):
        Detection(class_id=0, class_name="person", confidence=1.5, bounding_box=bb)


def test_detection_empty_class_name_invalid():
    bb = BoundingBox(0.0, 0.0, 50.0, 80.0)
    with pytest.raises(ValueError, match="class_name"):
        Detection(class_id=0, class_name="", confidence=0.5, bounding_box=bb)


# ---------------------------------------------------------------------------
# DetectionResult tests
# ---------------------------------------------------------------------------

def _make_detection(cls_id: int = 0, conf: float = 0.9) -> Detection:
    bb = BoundingBox(10.0, 20.0, 110.0, 120.0)
    return Detection(
        class_id=cls_id,
        class_name=COCO_CLASS_NAMES.get(cls_id, "unknown"),
        confidence=conf,
        bounding_box=bb,
    )


def test_detection_result_valid():
    result = DetectionResult(
        camera_id="CAM_01",
        frame_index=5,
        timestamp_seconds=0.5,
        detections=[_make_detection()],
        inference_time_ms=12.3,
    )
    assert result.detection_count == 1
    assert result.camera_id == "CAM_01"
    assert result.frame_index == 5


def test_detection_result_empty_detections_is_valid():
    """Zero detections is a valid result (no objects in frame)."""
    result = DetectionResult(
        camera_id="CAM_02",
        frame_index=0,
        timestamp_seconds=0.0,
        detections=[],
        inference_time_ms=8.0,
    )
    assert result.detection_count == 0


def test_detection_result_negative_frame_index_invalid():
    with pytest.raises(ValueError, match="frame_index"):
        DetectionResult(
            camera_id="CAM_01",
            frame_index=-1,
            timestamp_seconds=0.0,
            detections=[],
            inference_time_ms=5.0,
        )


def test_detection_result_preserves_metadata():
    """DetectionResult must carry through camera_id, frame_index, and timestamp."""
    result = DetectionResult(
        camera_id="CAM_TEST",
        frame_index=42,
        timestamp_seconds=4.2,
        detections=[_make_detection(cls_id=2)],
        inference_time_ms=15.0,
    )
    assert result.camera_id == "CAM_TEST"
    assert result.frame_index == 42
    assert result.timestamp_seconds == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# YOLODetector with mocked model (no weights downloaded)
# ---------------------------------------------------------------------------

def _make_frame_packet(frame_index: int = 0) -> FramePacket:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return FramePacket(
        camera_id="CAM_MOCK",
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10.0,
        frame=frame,
        source="mock_source.mp4",
    )


def _make_mock_box(class_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    """Create a mock YOLO box object matching the Ultralytics Boxes API."""
    import torch
    box = MagicMock()
    box.cls = [torch.tensor(float(class_id))]
    box.conf = [torch.tensor(conf)]
    box.xyxy = [torch.tensor([x1, y1, x2, y2])]
    return box


@pytest.fixture
def mocked_detector():
    """Return a YOLODetector with a completely mocked YOLO model."""
    with patch("aegis_ad.detection.detector.YOLODetector.__init__", lambda self, **kw: None):
        detector = object.__new__(YOLODetector)
    detector.model_path = "mock.pt"
    detector.confidence_threshold = 0.25
    detector.image_size = 640
    detector.device = "cpu"
    detector.target_class_ids = TARGET_CLASS_IDS
    detector._model = MagicMock()
    return detector


def _configure_mock_model(detector: YOLODetector, boxes: list) -> None:
    """Point detector._model() return value at given mock boxes."""
    mock_result = MagicMock()
    mock_result.boxes = boxes
    detector._model.return_value = [mock_result]


def test_detect_returns_person_detection(mocked_detector: YOLODetector):
    """Mocked inference with one person box produces one Detection."""
    mock_box = _make_mock_box(0, 0.92, 50.0, 30.0, 200.0, 400.0)
    _configure_mock_model(mocked_detector, [mock_box])

    packet = _make_frame_packet(frame_index=3)
    result = mocked_detector.detect(packet)

    assert result.camera_id == "CAM_MOCK"
    assert result.frame_index == 3
    assert result.timestamp_seconds == pytest.approx(0.3)
    assert result.detection_count == 1
    det = result.detections[0]
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.92)


def test_detect_filters_non_target_classes(mocked_detector: YOLODetector):
    """A class ID outside target set must be excluded from results."""
    mock_box = _make_mock_box(16, 0.88, 10.0, 10.0, 100.0, 100.0)  # 16 = dog (not a target)
    _configure_mock_model(mocked_detector, [mock_box])

    result = mocked_detector.detect(_make_frame_packet())
    assert result.detection_count == 0


def test_detect_empty_boxes_returns_empty_result(mocked_detector: YOLODetector):
    """Zero detections from the model produces a valid empty DetectionResult."""
    mock_result = MagicMock()
    mock_result.boxes = []
    mocked_detector._model.return_value = [mock_result]

    result = mocked_detector.detect(_make_frame_packet())
    assert result.detection_count == 0
    assert isinstance(result.detections, list)


def test_detect_multiple_classes(mocked_detector: YOLODetector):
    """Mixed-class output: person and car both retained, non-target excluded."""
    boxes = [
        _make_mock_box(0, 0.9, 0.0, 0.0, 100.0, 200.0),   # person
        _make_mock_box(2, 0.8, 200.0, 0.0, 400.0, 300.0),  # car
        _make_mock_box(15, 0.7, 10.0, 10.0, 50.0, 50.0),   # cat (excluded)
    ]
    _configure_mock_model(mocked_detector, boxes)

    result = mocked_detector.detect(_make_frame_packet())
    assert result.detection_count == 2
    names = {d.class_name for d in result.detections}
    assert names == {"person", "car"}


def test_detect_preserves_frame_metadata(mocked_detector: YOLODetector):
    """DetectionResult must preserve camera_id, frame_index, and timestamp from packet."""
    _configure_mock_model(mocked_detector, [])

    packet = FramePacket(
        camera_id="CAM_PRESERVE",
        frame_index=99,
        timestamp_seconds=9.9,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        source="test.mp4",
    )
    result = mocked_detector.detect(packet)
    assert result.camera_id == "CAM_PRESERVE"
    assert result.frame_index == 99
    assert result.timestamp_seconds == pytest.approx(9.9)


def test_detect_inference_error_raises_detection_error(mocked_detector: YOLODetector):
    """If the model raises during inference, a DetectionError is raised."""
    mocked_detector._model.side_effect = RuntimeError("CUDA OOM")
    with pytest.raises(DetectionError, match="inference failed"):
        mocked_detector.detect(_make_frame_packet())


def test_target_class_ids_contains_required_classes():
    """Confirm all six required COCO class IDs are present in TARGET_CLASS_IDS."""
    required = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
    assert required.issubset(TARGET_CLASS_IDS)


def test_coco_class_names_all_targets_named():
    """All target class IDs must have a non-empty name in COCO_CLASS_NAMES."""
    for cls_id in TARGET_CLASS_IDS:
        assert cls_id in COCO_CLASS_NAMES
        assert COCO_CLASS_NAMES[cls_id].strip()


# ---------------------------------------------------------------------------
# CLI configuration-wiring tests (no model weights or network access)
# ---------------------------------------------------------------------------

def _make_det_settings(detection_overrides: dict) -> MagicMock:
    """Return a mock Settings object whose raw dict contains the given detection section."""
    mock_settings = MagicMock()
    mock_settings.raw = {"detection": detection_overrides}
    return mock_settings


def _run_cli_with_mocks(
    args: list[str],
    detection_cfg: dict,
    *,
    # Optional real video path to pass; set False to skip the file-exists check
    source_path: str | None = None,
) -> tuple[MagicMock, int]:
    """
    Run run_cli with VideoSource, YOLODetector, and load_settings mocked.

    Returns (detector_init_mock, exit_code) so callers can inspect the kwargs
    YOLODetector was instantiated with.
    """
    from aegis_ad.detection.__main__ import run_cli

    detector_init_calls = []

    # A sentinel read_frames that yields nothing so the processing loop exits cleanly
    mock_video = MagicMock()
    mock_video.metadata.camera_id = "CAM_MOCK"
    mock_video.metadata.source_path = "mock.mp4"
    mock_video.metadata.width = 640
    mock_video.metadata.height = 480
    mock_video.metadata.fps = 25.0
    mock_video.read_frames.return_value = iter([])

    class _MockDetector:
        def __init__(self, **kwargs):
            detector_init_calls.append(kwargs)
            self.image_size = kwargs.get("image_size", 640)
            self.target_class_ids = kwargs.get("target_class_ids") or TARGET_CLASS_IDS

        def detect(self, packet):  # pragma: no cover – not exercised in these tests
            raise NotImplementedError

    with (
        patch("aegis_ad.detection.__main__.load_settings", return_value=_make_det_settings(detection_cfg)),
        patch("aegis_ad.detection.__main__.VideoSource", return_value=mock_video),
        patch("aegis_ad.detection.__main__.YOLODetector", side_effect=lambda **kw: _MockDetector(**kw)),
        patch("pathlib.Path.exists", return_value=True),
    ):
        exit_code = run_cli(args)

    init_kwargs = detector_init_calls[0] if detector_init_calls else {}
    mock_detector_call = MagicMock()
    mock_detector_call.kwargs = init_kwargs
    return mock_detector_call, exit_code


def test_cli_passes_image_size_from_config():
    """A non-default image_size in config must be passed to YOLODetector."""
    call, code = _run_cli_with_mocks(
        ["--source", "any.mp4", "--model", "mock.pt"],
        {"image_size": 320, "confidence_threshold": 0.25, "device": "cpu"},
    )
    assert code == 0
    assert call.kwargs["image_size"] == 320


def test_cli_passes_target_classes_from_config():
    """A custom target_classes list in config must reach YOLODetector as a frozenset."""
    call, code = _run_cli_with_mocks(
        ["--source", "any.mp4", "--model", "mock.pt"],
        {"target_classes": [0], "confidence_threshold": 0.25, "device": "cpu"},
    )
    assert code == 0
    assert call.kwargs["target_class_ids"] == frozenset({0})


def test_cli_uses_default_image_size_when_absent_from_config():
    """When image_size is absent from config, YOLODetector must receive the default 640."""
    call, code = _run_cli_with_mocks(
        ["--source", "any.mp4", "--model", "mock.pt"],
        {"confidence_threshold": 0.25, "device": "cpu"},  # no image_size key
    )
    assert code == 0
    assert call.kwargs["image_size"] == 640


def test_cli_passes_none_target_class_ids_when_absent_from_config():
    """When target_classes is absent from config, target_class_ids=None is passed,
    leaving YOLODetector to apply its own default TARGET_CLASS_IDS."""
    call, code = _run_cli_with_mocks(
        ["--source", "any.mp4", "--model", "mock.pt"],
        {"confidence_threshold": 0.25, "device": "cpu"},  # no target_classes key
    )
    assert code == 0
    assert call.kwargs["target_class_ids"] is None
