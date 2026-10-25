"""Data models for safety alerts in AEGIS-AD."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any
import uuid

from aegis_ad.events.models import SafetyEvent


class AlertError(Exception):
    """Raised when alert creation, validation, or processing fails."""


VALID_EVIDENCE_STATUSES = frozenset({
    "captured",
    "failed",
    "no_frame",
    "none",
})


def format_alert_title(event_type: str) -> str:
    """Format a clean, human-readable alert title from an event type."""
    if event_type == "restricted_area_entry":
        return "Restricted Area Entry"
    if event_type == "dwell_time_exceeded":
        return "Dwell Time Exceeded"
    return event_type.replace("_", " ").title()


def format_alert_message(event: SafetyEvent) -> str:
    """Format a concise, human-readable alert message in plain English."""
    zone_label = f"'{event.zone_name}'" if event.zone_name else f"zone {event.zone_id}"
    if event.event_type == "restricted_area_entry":
        return f"Person (track {event.track_id}) entered {zone_label} on camera {event.camera_id}."
    if event.event_type == "dwell_time_exceeded":
        dwell = event.details.get("dwell_duration_seconds")
        if dwell is not None and isinstance(dwell, (int, float)):
            return f"Person (track {event.track_id}) remained in {zone_label} on camera {event.camera_id} for {dwell:.1f}s."
        return f"Person (track {event.track_id}) exceeded dwell limit in {zone_label} on camera {event.camera_id}."
    return f"Safety event '{event.event_type}' detected for track {event.track_id} on camera {event.camera_id}."


@dataclass(frozen=True, slots=True)
class Alert:
    """Structured record of an alert generated from a SafetyEvent."""

    alert_id: str
    event_id: str
    event_type: str
    severity: str
    camera_id: str
    zone_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    title: str
    message: str
    evidence_path: str | None = None
    evidence_status: str = "none"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.alert_id or not isinstance(self.alert_id, str):
            raise ValueError("alert_id must be a non-empty string.")
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValueError("event_id must be a non-empty string.")
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("event_type must be a non-empty string.")
        if not self.severity or not isinstance(self.severity, str):
            raise ValueError("severity must be a non-empty string.")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if not self.zone_id or not isinstance(self.zone_id, str):
            raise ValueError("zone_id must be a non-empty string.")
        if self.track_id < 1:
            raise ValueError(f"track_id must be positive, got {self.track_id}.")
        if self.frame_index < 0:
            raise ValueError(f"frame_index cannot be negative, got {self.frame_index}.")
        if self.timestamp_seconds < 0.0 or not math.isfinite(self.timestamp_seconds):
            raise ValueError(
                f"timestamp_seconds must be a non-negative finite number, got {self.timestamp_seconds}."
            )
        if not self.title or not isinstance(self.title, str):
            raise ValueError("title must be a non-empty string.")
        if not self.message or not isinstance(self.message, str):
            raise ValueError("message must be a non-empty string.")
        if self.evidence_status not in VALID_EVIDENCE_STATUSES:
            raise ValueError(
                f"evidence_status must be one of {sorted(VALID_EVIDENCE_STATUSES)}, got '{self.evidence_status}'."
            )

    @classmethod
    def from_event(
        cls,
        event: SafetyEvent,
        evidence_path: str | None = None,
        evidence_status: str = "none",
        alert_id: str | None = None,
        created_at: str | None = None,
    ) -> "Alert":
        """Factory creating an Alert from a SafetyEvent and optional evidence metadata."""
        aid = alert_id or str(uuid.uuid4())
        cat = created_at or datetime.now(timezone.utc).isoformat()
        return cls(
            alert_id=aid,
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            camera_id=event.camera_id,
            zone_id=event.zone_id,
            track_id=event.track_id,
            frame_index=event.frame_index,
            timestamp_seconds=event.timestamp_seconds,
            title=format_alert_title(event.event_type),
            message=format_alert_message(event),
            evidence_path=evidence_path,
            evidence_status=evidence_status,
            created_at=cat,
        )
