"""Event engine for safety rules and state transitions."""

from __future__ import annotations

import logging
import math
from typing import Any
import uuid

from aegis_ad.events.models import EventConfigurationError, RestrictedZone, SafetyEvent
from aegis_ad.tracking.models import TrackedFrame, TrackedObject

logger = logging.getLogger(__name__)


class _TrackZoneState:
    """Internal state tracking for a single track within a single zone."""

    __slots__ = (
        "track_id",
        "zone_id",
        "camera_id",
        "has_been_outside",
        "is_inside",
        "inside_since_timestamp",
        "last_seen_timestamp",
        "last_seen_frame_index",
        "dwell_event_emitted",
    )

    def __init__(
        self,
        track_id: int,
        zone_id: str,
        camera_id: str,
    ) -> None:
        self.track_id = track_id
        self.zone_id = zone_id
        self.camera_id = camera_id
        self.has_been_outside: bool = False
        self.is_inside: bool = False
        self.inside_since_timestamp: float | None = None
        self.last_seen_timestamp: float = 0.0
        self.last_seen_frame_index: int = 0
        self.dwell_event_emitted: bool = False


class EventEngine:
    """Evaluates safety rules (restricted zone entry, dwell time) on tracked frames."""

    def __init__(
        self,
        zones: list[RestrictedZone] | None = None,
        dwell_threshold_seconds: float = 30.0,
        absence_grace_seconds: float = 1.0,
        person_class_id: int = 0,
        enabled: bool = True,
    ) -> None:
        """Initialise the event engine with configured zones and rule parameters.

        Args:
            zones: Configured restricted zones.
            dwell_threshold_seconds: Seconds a person must continuously dwell in a zone.
            absence_grace_seconds: Maximum tracking gap tolerated before resetting dwell.
            person_class_id: Class ID considered for person rules (COCO class 0).
            enabled: Master switch for event rule evaluation.
        """
        if dwell_threshold_seconds < 0.0 or not math.isfinite(dwell_threshold_seconds):
            raise EventConfigurationError(
                f"dwell_threshold_seconds must be a non-negative finite number, got {dwell_threshold_seconds}."
            )
        if absence_grace_seconds < 0.0 or not math.isfinite(absence_grace_seconds):
            raise EventConfigurationError(
                f"absence_grace_seconds must be a non-negative finite number, got {absence_grace_seconds}."
            )
        if person_class_id < 0:
            raise EventConfigurationError(
                f"person_class_id cannot be negative, got {person_class_id}."
            )

        self.dwell_threshold_seconds = float(dwell_threshold_seconds)
        self.absence_grace_seconds = float(absence_grace_seconds)
        self.person_class_id = int(person_class_id)
        self.enabled = bool(enabled)

        # Index zones by camera_id
        self._zones_by_camera: dict[str, list[RestrictedZone]] = {}
        for zone in (zones or []):
            if not isinstance(zone, RestrictedZone):
                raise EventConfigurationError(
                    f"All zones must be RestrictedZone instances, got {type(zone).__name__}."
                )
            self._zones_by_camera.setdefault(zone.camera_id, []).append(zone)

        # State mapping: (camera_id, zone_id, track_id) -> _TrackZoneState
        self._states: dict[tuple[str, str, int], _TrackZoneState] = {}

    @property
    def zones(self) -> list[RestrictedZone]:
        """Return all registered zones."""
        result: list[RestrictedZone] = []
        for zone_list in self._zones_by_camera.values():
            result.extend(zone_list)
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EventEngine":
        """Instantiate EventEngine from an events configuration dictionary."""
        enabled = config.get("enabled", True)
        dwell_threshold = float(config.get("dwell_threshold_seconds", 30.0))
        absence_grace = float(config.get("absence_grace_seconds", 1.0))
        person_class_id = int(config.get("person_class_id", 0))

        raw_zones = config.get("zones", [])
        zones: list[RestrictedZone] = []
        for z in raw_zones:
            if not isinstance(z, dict):
                continue
            zone_id = str(z.get("zone_id", ""))
            name = str(z.get("name", zone_id))
            camera_id = str(z.get("camera_id", ""))
            raw_poly = z.get("polygon", [])
            poly = tuple(tuple(float(coord) for coord in pt) for pt in raw_poly)
            zones.append(
                RestrictedZone(
                    zone_id=zone_id,
                    name=name,
                    camera_id=camera_id,
                    polygon=poly,
                )
            )

        return cls(
            zones=zones,
            dwell_threshold_seconds=dwell_threshold,
            absence_grace_seconds=absence_grace,
            person_class_id=person_class_id,
            enabled=enabled,
        )
        """Return all registered zones."""
        result: list[RestrictedZone] = []
        for zone_list in self._zones_by_camera.values():
            result.extend(zone_list)
        return result

    def reset(self, camera_id: str | None = None) -> None:
        """Reset event state. If camera_id is given, reset only for that camera."""
        if camera_id is None:
            self._states.clear()
            logger.debug("EventEngine: reset all state.")
        else:
            to_remove = [k for k in self._states if k[0] == camera_id]
            for k in to_remove:
                del self._states[k]
            logger.debug("EventEngine: reset state for camera '%s'.", camera_id)

    def process_frame(
        self,
        frame: TrackedFrame,
        frame_width: int,
        frame_height: int,
    ) -> list[SafetyEvent]:
        """Evaluate event rules on a single TrackedFrame.

        Args:
            frame: Tracked frame containing per-camera tracked objects.
            frame_width: Frame pixel width (used to normalize bounding boxes).
            frame_height: Frame pixel height (used to normalize bounding boxes).

        Returns:
            List of SafetyEvent instances emitted on this frame.
        """
        if not self.enabled:
            return []

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                f"Invalid frame dimensions: width={frame_width}, height={frame_height}."
            )

        zones = self._zones_by_camera.get(frame.camera_id, [])
        if not zones:
            return []

        # Filter qualifying person tracks
        qualifying_tracks: list[TrackedObject] = [
            obj for obj in frame.tracked_objects if obj.class_id == self.person_class_id
        ]
        current_track_ids = {obj.track_id for obj in qualifying_tracks}

        # Check tracking gaps for tracks not present in the current frame
        current_time = frame.timestamp_seconds
        for (cam_id, zone_id, trk_id), state in list(self._states.items()):
            if cam_id != frame.camera_id:
                continue
            if trk_id not in current_track_ids:
                gap = current_time - state.last_seen_timestamp
                if gap > (self.absence_grace_seconds + 1e-6):
                    # Missing interval exceeded tolerance: expire presence and outside history
                    state.is_inside = False
                    state.inside_since_timestamp = None
                    state.dwell_event_emitted = False
                    state.has_been_outside = False

        events: list[SafetyEvent] = []

        # Evaluate each zone against each qualifying track
        for zone in zones:
            for obj in qualifying_tracks:
                # Reference point: bottom-center of bounding box
                bc_x = (obj.x1 + obj.x2) / 2.0
                bc_y = obj.y2
                norm_x = bc_x / float(frame_width)
                norm_y = bc_y / float(frame_height)

                state_key = (frame.camera_id, zone.zone_id, obj.track_id)
                state = self._states.get(state_key)
                if state is None:
                    state = _TrackZoneState(obj.track_id, zone.zone_id, frame.camera_id)
                    self._states[state_key] = state

                is_inside = zone.contains_point((norm_x, norm_y))

                if is_inside:
                    time_since_last_seen = current_time - state.last_seen_timestamp
                    is_continuous = state.is_inside and (time_since_last_seen <= (self.absence_grace_seconds + 1e-6))

                    if is_continuous:
                        # Continuing existing presence episode
                        state.last_seen_timestamp = current_time
                        state.last_seen_frame_index = frame.frame_index

                        inside_since = (
                            state.inside_since_timestamp
                            if state.inside_since_timestamp is not None
                            else current_time
                        )
                        dwell_duration = current_time - inside_since
                        if dwell_duration >= self.dwell_threshold_seconds and not state.dwell_event_emitted:
                            state.dwell_event_emitted = True
                            event = SafetyEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="dwell_time_exceeded",
                                camera_id=frame.camera_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                track_id=obj.track_id,
                                frame_index=frame.frame_index,
                                timestamp_seconds=current_time,
                                severity="high",
                                details={
                                    "dwell_duration_seconds": dwell_duration,
                                    "dwell_threshold_seconds": self.dwell_threshold_seconds,
                                    "reference_point": (norm_x, norm_y),
                                },
                            )
                            events.append(event)
                            logger.info(
                                "[%s] Dwell time exceeded: track %d in '%s' (%.1fs)",
                                frame.camera_id, obj.track_id, zone.name, dwell_duration
                            )
                    else:
                        # Entering zone or re-entering after gap/exit
                        if state.has_been_outside:
                            # Observed transition from outside to inside!
                            event = SafetyEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="restricted_area_entry",
                                camera_id=frame.camera_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                track_id=obj.track_id,
                                frame_index=frame.frame_index,
                                timestamp_seconds=current_time,
                                severity="medium",
                                details={
                                    "reference_point": (norm_x, norm_y),
                                },
                            )
                            events.append(event)
                            logger.info(
                                "[%s] Restricted area entry: track %d entered '%s'",
                                frame.camera_id, obj.track_id, zone.name
                            )

                        # Start or restart dwell episode
                        state.is_inside = True
                        state.inside_since_timestamp = current_time
                        state.last_seen_timestamp = current_time
                        state.last_seen_frame_index = frame.frame_index
                        state.dwell_event_emitted = False
                        state.has_been_outside = False

                        # Immediate dwell check if threshold is 0.0
                        if self.dwell_threshold_seconds <= 0.0 and not state.dwell_event_emitted:
                            state.dwell_event_emitted = True
                            events.append(
                                SafetyEvent(
                                    event_id=str(uuid.uuid4()),
                                    event_type="dwell_time_exceeded",
                                    camera_id=frame.camera_id,
                                    zone_id=zone.zone_id,
                                    zone_name=zone.name,
                                    track_id=obj.track_id,
                                    frame_index=frame.frame_index,
                                    timestamp_seconds=current_time,
                                    severity="high",
                                    details={
                                        "dwell_duration_seconds": 0.0,
                                        "dwell_threshold_seconds": 0.0,
                                        "reference_point": (norm_x, norm_y),
                                    },
                                )
                            )
                else:
                    # Outside the zone
                    state.is_inside = False
                    state.inside_since_timestamp = None
                    state.dwell_event_emitted = False
                    state.has_been_outside = True
                    state.last_seen_timestamp = current_time
                    state.last_seen_frame_index = frame.frame_index

        return events
