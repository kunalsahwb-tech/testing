"""Tests for EvidenceStore and snapshot capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest

from aegis_ad.evidence.store import EvidenceStore, _sanitize_filename_component
from aegis_ad.events.models import SafetyEvent


def _make_event(event_id: str = "evt-snapshot-123") -> SafetyEvent:
    return SafetyEvent(
        event_id=event_id,
        event_type="restricted_area_entry",
        camera_id="CAM_TEST_01",
        zone_id="z1",
        zone_name="Perimeter",
        track_id=5,
        frame_index=42,
        timestamp_seconds=4.2,
        severity="medium",
    )


def test_sanitize_filename_component():
    assert _sanitize_filename_component("CAM/01:test*") == "CAM_01_test_"
    assert _sanitize_filename_component("   ") == "unknown"


def test_save_snapshot_success(tmp_path: Path):
    store = EvidenceStore(evidence_dir=tmp_path)
    event = _make_event()

    # Synthetic BGR image (100x100 3-channel uint8)
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    path_str, status = store.save_snapshot(event, frame)
    assert status == "captured"
    assert path_str is not None

    # Check file exists and can be decoded back
    saved_file = Path(path_str)
    if not saved_file.is_absolute():
        saved_file = Path(path_str)  # could be relative to root
    # Since we passed tmp_path, let's verify file in tmp_path
    files = list(tmp_path.glob("*.jpg"))
    assert len(files) == 1
    read_img = cv2.imread(str(files[0]))
    assert read_img is not None
    assert read_img.shape == (100, 100, 3)


def test_save_snapshot_no_frame(tmp_path: Path):
    store = EvidenceStore(evidence_dir=tmp_path)
    event = _make_event()

    path_str, status = store.save_snapshot(event, None)
    assert status == "no_frame"
    assert path_str is None
    assert len(list(tmp_path.glob("*.jpg"))) == 0


def test_save_snapshot_invalid_frame_array(tmp_path: Path):
    store = EvidenceStore(evidence_dir=tmp_path)
    event = _make_event()

    empty_frame = np.empty((0,), dtype=np.uint8)
    path_str, status = store.save_snapshot(event, empty_frame)
    assert status == "failed"
    assert path_str is None
    assert len(list(tmp_path.glob("*.jpg"))) == 0


def test_save_snapshot_avoids_overwriting_existing(tmp_path: Path):
    store = EvidenceStore(evidence_dir=tmp_path)
    event = _make_event()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    path1, status1 = store.save_snapshot(event, frame)
    path2, status2 = store.save_snapshot(event, frame)

    assert status1 == "captured"
    assert status2 == "captured"
    assert path1 != path2

    files = list(tmp_path.glob("*.jpg"))
    assert len(files) == 2


def test_save_snapshot_write_failure_returns_failed(tmp_path: Path):
    store = EvidenceStore(evidence_dir=tmp_path)
    event = _make_event()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    with patch("cv2.imwrite", return_value=False):
        path_str, status = store.save_snapshot(event, frame)
        assert status == "failed"
        assert path_str is None
