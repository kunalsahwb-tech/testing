"""Orchestration service connecting safety events, alerts, and evidence capture."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aegis_ad.alerts.models import Alert
from aegis_ad.alerts.repository import AlertRepository
from aegis_ad.evidence.store import EvidenceStore
from aegis_ad.events.models import SafetyEvent

logger = logging.getLogger(__name__)


class AlertService:
    """Orchestrates alert creation, evidence capture, and persistent storage."""

    def __init__(
        self,
        repository: AlertRepository,
        evidence_store: EvidenceStore,
    ) -> None:
        """Initialise the alert service.

        Args:
            repository: Repository for alert persistence.
            evidence_store: Store for saving and managing evidence files.
        """
        self.repository = repository
        self.evidence_store = evidence_store

    def process_event(
        self,
        event: SafetyEvent,
        frame: np.ndarray | None = None,
    ) -> Alert:
        """Process a SafetyEvent by capturing evidence snapshot and persisting an Alert.

        Idempotency: If an alert for event.event_id already exists, the existing alert
        is returned and no duplicate evidence or alert is created.

        Resilience: If evidence capture fails or raises an error, the alert is still
        persisted with evidence_status='failed'. A valid event is never lost due to
        evidence capture issues.

        Args:
            event: The safety event emitted by the EventEngine.
            frame: Optional BGR image array corresponding to the event frame.

        Returns:
            The persisted Alert record.
        """
        # Idempotency check: don't recreate or recapture for existing event
        existing = self.repository.get_by_event_id(event.event_id)
        if existing is not None:
            logger.debug(
                "[%s] Alert already exists for event %s (alert_id=%s).",
                event.camera_id, event.event_id, existing.alert_id
            )
            return existing

        # Attempt snapshot capture
        evidence_path: str | None = None
        evidence_status: str = "none"

        try:
            evidence_path, evidence_status = self.evidence_store.save_snapshot(event, frame)
        except Exception as exc:
            logger.error(
                "[%s] Unexpected error capturing evidence for event %s: %s",
                event.camera_id, event.event_id, exc
            )
            evidence_path = None
            evidence_status = "failed"

        # Create alert with evidence metadata
        alert = Alert.from_event(
            event=event,
            evidence_path=evidence_path,
            evidence_status=evidence_status,
        )

        # Persist alert record
        saved = self.repository.save(alert)
        logger.info(
            "[%s] Alert created: [%s] %s (evidence: %s)",
            saved.camera_id, saved.alert_id[:8], saved.title, saved.evidence_status
        )
        return saved

    def get_alert(self, alert_id: str) -> Alert | None:
        """Retrieve an alert by its ID."""
        return self.repository.get_by_id(alert_id)

    def list_alerts(
        self,
        limit: int = 50,
        camera_id: str | None = None,
        event_type: str | None = None,
    ) -> list[Alert]:
        """List recent alerts with optional filters."""
        return self.repository.list_recent(limit=limit, camera_id=camera_id, event_type=event_type)
