"""Persistent storage for alert records using SQLite."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
from typing import Any

from aegis_ad.alerts.models import Alert, AlertError
from aegis_ad.settings import get_project_root

logger = logging.getLogger(__name__)


class AlertRepository:
    """SQLite-backed repository for persistent Alert records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialise the alert repository.

        Args:
            db_path: Path to the SQLite database file. Defaults to data/alerts.db.
        """
        if db_path is None:
            self.db_path = get_project_root() / "data" / "alerts.db"
        else:
            self.db_path = Path(db_path).resolve()

        self._ensure_directory()
        self.init_schema()

    def _ensure_directory(self) -> None:
        """Ensure parent directory for the database exists."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create database directory '%s': %s", self.db_path.parent, exc)

    def _get_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for row access."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Create the alerts schema and indices if they do not exist."""
        self._ensure_directory()
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_seconds REAL NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    evidence_path TEXT,
                    evidence_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp_seconds)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alerts(event_id)"
            )
            conn.commit()

    def save(self, alert: Alert) -> Alert:
        """Persist an Alert record idempotently.

        If an alert with the same event_id already exists, returns the existing record.
        """
        existing = self.get_by_event_id(alert.event_id)
        if existing is not None:
            logger.debug("Alert for event '%s' already exists (alert_id='%s').", alert.event_id, existing.alert_id)
            return existing

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, event_id, event_type, severity,
                        camera_id, zone_id, track_id, frame_index,
                        timestamp_seconds, title, message,
                        evidence_path, evidence_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert.alert_id,
                        alert.event_id,
                        alert.event_type,
                        alert.severity,
                        alert.camera_id,
                        alert.zone_id,
                        alert.track_id,
                        alert.frame_index,
                        alert.timestamp_seconds,
                        alert.title,
                        alert.message,
                        alert.evidence_path,
                        alert.evidence_status,
                        alert.created_at,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # Concurrent or retry insert caught by DB constraint
                found = self.get_by_event_id(alert.event_id)
                if found is not None:
                    return found
                raise
            except sqlite3.Error as exc:
                raise AlertError(f"Database error while saving alert '{alert.alert_id}': {exc}") from exc

        return alert

    def get_by_id(self, alert_id: str) -> Alert | None:
        """Retrieve an Alert by its unique alert_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_alert(row)

    def get_by_event_id(self, event_id: str) -> Alert | None:
        """Retrieve an Alert by its associated event_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM alerts WHERE event_id = ?",
                (event_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_alert(row)

    def list_recent(
        self,
        limit: int = 50,
        camera_id: str | None = None,
        event_type: str | None = None,
    ) -> list[Alert]:
        """List alerts ordered from newest to oldest by event timestamp.

        Args:
            limit: Maximum number of alerts to return.
            camera_id: Optional camera filter.
            event_type: Optional event_type filter.
        """
        query = "SELECT * FROM alerts"
        conditions: list[str] = []
        params: list[Any] = []

        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp_seconds DESC, frame_index DESC LIMIT ?"
        params.append(max(1, limit))

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        """Convert a sqlite3.Row to an Alert instance."""
        return Alert(
            alert_id=row["alert_id"],
            event_id=row["event_id"],
            event_type=row["event_type"],
            severity=row["severity"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            track_id=row["track_id"],
            frame_index=row["frame_index"],
            timestamp_seconds=row["timestamp_seconds"],
            title=row["title"],
            message=row["message"],
            evidence_path=row["evidence_path"],
            evidence_status=row["evidence_status"],
            created_at=row["created_at"],
        )
