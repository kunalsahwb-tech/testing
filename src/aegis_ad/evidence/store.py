"""Evidence capture and snapshot storage for AEGIS-AD safety events."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aegis_ad.events.models import SafetyEvent
from aegis_ad.settings import get_project_root

logger = logging.getLogger(__name__)


def _sanitize_filename_component(name: str) -> str:
    """Sanitize arbitrary strings to safe alphanumeric filesystem components."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    return cleaned or "unknown"


class EvidenceStore:
    """Manages snapshot evidence capture and persistent file referencing."""

    def __init__(
        self,
        evidence_dir: str | Path | None = None,
        image_format: str = "jpg",
    ) -> None:
        """Initialise the evidence store.

        Args:
            evidence_dir: Base directory for storing snapshots. Defaults to data/evidence/.
            image_format: Image file extension / format (default 'jpg').
        """
        if evidence_dir is None:
            self.evidence_dir = get_project_root() / "data" / "evidence"
        else:
            self.evidence_dir = Path(evidence_dir).resolve()

        self.image_format = image_format.lstrip(".").lower()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create evidence directory if it does not already exist."""
        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create evidence directory '%s': %s", self.evidence_dir, exc)

    def save_snapshot(
        self,
        event: SafetyEvent,
        frame: np.ndarray | None,
    ) -> tuple[str | None, str]:
        """Save a snapshot of the event frame.

        Args:
            event: The triggering SafetyEvent.
            frame: The BGR numpy array of the frame (from FramePacket).

        Returns:
            Tuple of (evidence_path, evidence_status).
            evidence_status is one of: 'captured', 'failed', 'no_frame'.
            evidence_path is a portable relative path string if captured, else None.
        """
        if frame is None:
            logger.debug(
                "[%s] No frame image provided for event %s (frame %d).",
                event.camera_id, event.event_id, event.frame_index
            )
            return None, "no_frame"

        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
            logger.warning(
                "[%s] Invalid frame array for event %s (frame %d).",
                event.camera_id, event.event_id, event.frame_index
            )
            return None, "failed"

        self._ensure_directory()

        safe_cam = _sanitize_filename_component(event.camera_id)
        safe_type = _sanitize_filename_component(event.event_type)
        safe_eid = _sanitize_filename_component(event.event_id[:8])
        base_name = f"{safe_cam}_{safe_type}_{safe_eid}_{event.frame_index:06d}.{self.image_format}"

        target_file = self.evidence_dir / base_name
        counter = 1
        while target_file.exists() and counter <= 1000:
            target_file = self.evidence_dir / f"{safe_cam}_{safe_type}_{safe_eid}_{event.frame_index:06d}_{counter}.{self.image_format}"
            counter += 1

        try:
            success = cv2.imwrite(str(target_file), frame)
            if not success:
                logger.warning("cv2.imwrite returned False for '%s'", target_file)
                return None, "failed"
        except Exception as exc:
            logger.warning("Failed to encode or write evidence snapshot '%s': %s", target_file, exc)
            return None, "failed"

        # Construct a clean, portable path relative to project root if possible
        try:
            root = get_project_root()
            rel_path = str(target_file.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel_path = str(target_file).replace("\\", "/")

        logger.info("[%s] Evidence snapshot saved: %s", event.camera_id, rel_path)
        return rel_path, "captured"
