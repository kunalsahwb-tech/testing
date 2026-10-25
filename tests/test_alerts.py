"""Tests for alerts, persistence, and evidence handling."""

from __future__ import annotations

import pytest

from aegis_ad.alerts.models import (
    Alert,
    AlertError,
    format_alert_message,
    format_alert_title,
)
from aegis_ad.events.models import SafetyEvent


def _make_sample_event(
    event_id: str = "evt-123",
    event_type: str = "restricted_area_entry",
    camera_id: str = "CAM_01",
    zone_id: str = "zone_01",
    zone_name: str = "Loading Dock",
    track_id: int = 42,
    frame_index: int = 100,
    timestamp_seconds: float = 10.0,
    severity: str = "high",
    details: dict | None = None,
) -> SafetyEvent:
    return SafetyEvent(
        event_id=event_id,
        event_type=event_type,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_name=zone_name,
        track_id=track_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        severity=severity,
        details=details or {},
    )


def test_alert_creation_from_event():
    event = _make_sample_event()
    alert = Alert.from_event(
        event=event,
        evidence_path="data/evidence/test.jpg",
        evidence_status="captured",
    )
    assert alert.event_id == "evt-123"
    assert alert.event_type == "restricted_area_entry"
    assert alert.title == "Restricted Area Entry"
    assert "Person (track 42) entered 'Loading Dock'" in alert.message
    assert alert.camera_id == "CAM_01"
    assert alert.zone_id == "zone_01"
    assert alert.track_id == 42
    assert alert.frame_index == 100
    assert alert.timestamp_seconds == 10.0
    assert alert.severity == "high"
    assert alert.evidence_path == "data/evidence/test.jpg"
    assert alert.evidence_status == "captured"
    assert alert.created_at is not None


def test_alert_dwell_exceeded_formatting():
    event = _make_sample_event(
        event_type="dwell_time_exceeded",
        details={"dwell_duration_seconds": 32.5},
    )
    alert = Alert.from_event(event)
    assert alert.title == "Dwell Time Exceeded"
    assert "remained in 'Loading Dock' on camera CAM_01 for 32.5s" in alert.message


def test_alert_validation_invalid_status_raises():
    with pytest.raises(ValueError, match="evidence_status"):
        Alert(
            alert_id="a1",
            event_id="e1",
            event_type="restricted_area_entry",
            severity="medium",
            camera_id="CAM_01",
            zone_id="z1",
            track_id=1,
            frame_index=0,
            timestamp_seconds=0.0,
            title="Title",
            message="Message",
            evidence_status="invalid_status",
        )


def test_alert_validation_negative_track_id_raises():
    with pytest.raises(ValueError, match="track_id"):
        Alert(
            alert_id="a1",
            event_id="e1",
            event_type="restricted_area_entry",
            severity="medium",
            camera_id="CAM_01",
            zone_id="z1",
            track_id=-1,
            frame_index=0,
            timestamp_seconds=0.0,
            title="Title",
            message="Message",
        )


# ---------------------------------------------------------------------------
# AlertRepository Tests (SQLite in-memory or tmp_path)
# ---------------------------------------------------------------------------

from pathlib import Path
from aegis_ad.alerts.repository import AlertRepository


def test_repository_save_and_retrieve(tmp_path: Path):
    db_file = tmp_path / "test_alerts.db"
    repo = AlertRepository(db_path=db_file)

    event = _make_sample_event(event_id="evt-save-1")
    alert = Alert.from_event(
        event=event,
        evidence_path="data/evidence/snap1.jpg",
        evidence_status="captured",
    )

    saved = repo.save(alert)
    assert saved.alert_id == alert.alert_id

    # Retrieve by alert_id
    retrieved = repo.get_by_id(alert.alert_id)
    assert retrieved is not None
    assert retrieved.alert_id == alert.alert_id
    assert retrieved.event_id == "evt-save-1"
    assert retrieved.evidence_path == "data/evidence/snap1.jpg"
    assert retrieved.evidence_status == "captured"

    # Retrieve by event_id
    by_evt = repo.get_by_event_id("evt-save-1")
    assert by_evt is not None
    assert by_evt.alert_id == alert.alert_id


def test_repository_get_non_existent_returns_none(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "test_empty.db")
    assert repo.get_by_id("does-not-exist") is None
    assert repo.get_by_event_id("does-not-exist") is None


def test_repository_idempotency_prevents_duplicate_events(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "test_idempotency.db")
    event = _make_sample_event(event_id="evt-duplicate")

    alert1 = Alert.from_event(event=event, alert_id="alert-1")
    alert2 = Alert.from_event(event=event, alert_id="alert-2")

    saved1 = repo.save(alert1)
    saved2 = repo.save(alert2)

    # Must return the original alert and not create a second row
    assert saved1.alert_id == "alert-1"
    assert saved2.alert_id == "alert-1"

    recent = repo.list_recent()
    assert len(recent) == 1
    assert recent[0].alert_id == "alert-1"


def test_repository_list_recent_ordering_and_filters(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "test_filters.db")

    ev1 = _make_sample_event(event_id="e1", camera_id="CAM_01", event_type="restricted_area_entry", timestamp_seconds=1.0)
    ev2 = _make_sample_event(event_id="e2", camera_id="CAM_02", event_type="restricted_area_entry", timestamp_seconds=5.0)
    ev3 = _make_sample_event(event_id="e3", camera_id="CAM_01", event_type="dwell_time_exceeded", timestamp_seconds=10.0)

    repo.save(Alert.from_event(ev1, alert_id="a1"))
    repo.save(Alert.from_event(ev2, alert_id="a2"))
    repo.save(Alert.from_event(ev3, alert_id="a3"))

    # Ordered newest first (timestamp 10.0 -> 5.0 -> 1.0)
    all_recent = repo.list_recent()
    assert [a.alert_id for a in all_recent] == ["a3", "a2", "a1"]

    # Filter by camera_id
    cam1_recent = repo.list_recent(camera_id="CAM_01")
    assert [a.alert_id for a in cam1_recent] == ["a3", "a1"]

    # Filter by event_type
    dwell_recent = repo.list_recent(event_type="dwell_time_exceeded")
    assert [a.alert_id for a in dwell_recent] == ["a3"]

    # Limit check
    limited = repo.list_recent(limit=2)
    assert len(limited) == 2
    assert [a.alert_id for a in limited] == ["a3", "a2"]


# ---------------------------------------------------------------------------
# AlertService Orchestration Tests
# ---------------------------------------------------------------------------

import numpy as np
from aegis_ad.alerts.service import AlertService
from aegis_ad.evidence.store import EvidenceStore


def test_alert_service_process_event_with_valid_frame(tmp_path: Path):
    db_file = tmp_path / "alerts.db"
    evidence_dir = tmp_path / "evidence"
    repo = AlertRepository(db_path=db_file)
    store = EvidenceStore(evidence_dir=evidence_dir)
    service = AlertService(repository=repo, evidence_store=store)

    event = _make_sample_event(event_id="evt-live-1", camera_id="CAM_01")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    alert = service.process_event(event=event, frame=frame)

    assert alert.event_id == "evt-live-1"
    assert alert.evidence_status == "captured"
    assert alert.evidence_path is not None
    assert Path(alert.evidence_path).is_file()

    # Retrieval from service
    retrieved = service.get_alert(alert.alert_id)
    assert retrieved is not None
    assert retrieved.alert_id == alert.alert_id
    assert retrieved.evidence_status == "captured"


def test_alert_service_process_event_without_frame(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "alerts.db")
    store = EvidenceStore(evidence_dir=tmp_path / "evidence")
    service = AlertService(repository=repo, evidence_store=store)

    event = _make_sample_event(event_id="evt-no-frame")
    alert = service.process_event(event=event, frame=None)

    assert alert.event_id == "evt-no-frame"
    assert alert.evidence_status == "no_frame"
    assert alert.evidence_path is None

    # Alert is still persisted
    saved = repo.get_by_event_id("evt-no-frame")
    assert saved is not None
    assert saved.evidence_status == "no_frame"


def test_alert_service_resilience_on_evidence_store_failure(tmp_path: Path, monkeypatch):
    repo = AlertRepository(db_path=tmp_path / "alerts.db")
    store = EvidenceStore(evidence_dir=tmp_path / "evidence")
    service = AlertService(repository=repo, evidence_store=store)

    def _failing_save(*args, **kwargs):
        raise RuntimeError("Simulated disk error or write permission failure")

    monkeypatch.setattr(store, "save_snapshot", _failing_save)

    event = _make_sample_event(event_id="evt-fail-disk")
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    # Must NOT raise exception
    alert = service.process_event(event=event, frame=frame)

    assert alert.event_id == "evt-fail-disk"
    assert alert.evidence_status == "failed"
    assert alert.evidence_path is None

    # Confirm it was saved
    saved = repo.get_by_event_id("evt-fail-disk")
    assert saved is not None
    assert saved.evidence_status == "failed"


def test_alert_service_idempotency_returns_existing(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "alerts.db")
    store = EvidenceStore(evidence_dir=tmp_path / "evidence")
    service = AlertService(repository=repo, evidence_store=store)

    event = _make_sample_event(event_id="evt-repeat")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    alert1 = service.process_event(event=event, frame=frame)
    alert2 = service.process_event(event=event, frame=frame)

    assert alert1.alert_id == alert2.alert_id
    assert alert1.created_at == alert2.created_at

    # Only 1 file should exist in evidence dir
    files = list((tmp_path / "evidence").glob("*.jpg"))
    assert len(files) == 1


def test_alert_service_list_alerts(tmp_path: Path):
    repo = AlertRepository(db_path=tmp_path / "alerts.db")
    store = EvidenceStore(evidence_dir=tmp_path / "evidence")
    service = AlertService(repository=repo, evidence_store=store)

    ev1 = _make_sample_event(event_id="e1", camera_id="CAM_A", timestamp_seconds=1.0)
    ev2 = _make_sample_event(event_id="e2", camera_id="CAM_B", timestamp_seconds=2.0)

    service.process_event(ev1)
    service.process_event(ev2)

    alerts = service.list_alerts()
    assert len(alerts) == 2
    assert alerts[0].event_id == "e2"

    filtered = service.list_alerts(camera_id="CAM_A")
    assert len(filtered) == 1
    assert filtered[0].camera_id == "CAM_A"

