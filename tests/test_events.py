"""Tests for event rules, restricted zones, and safety event engine."""

from __future__ import annotations

import pytest

from aegis_ad.events.engine import EventEngine
from aegis_ad.events.models import (
    EventConfigurationError,
    RestrictedZone,
    SafetyEvent,
    point_in_polygon,
)
from aegis_ad.tracking.models import TrackedFrame, TrackedObject


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_zone(
    zone_id: str = "zone_01",
    name: str = "Restricted Area",
    camera_id: str = "CAM_01",
    polygon: list[tuple[float, float]] | None = None,
) -> RestrictedZone:
    # Default: rectangle from (0.5, 0.5) to (1.0, 1.0)
    poly = polygon or [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]
    return RestrictedZone(
        zone_id=zone_id,
        name=name,
        camera_id=camera_id,
        polygon=tuple(poly),
    )


def _make_tracked_obj(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    # In a 1000x1000 frame:
    # bottom-center = ((x1+x2)/2, y2)
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _make_frame(
    camera_id: str = "CAM_01",
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    tracked_objects: list[TrackedObject] | None = None,
) -> TrackedFrame:
    return TrackedFrame(
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        tracked_objects=tracked_objects if tracked_objects is not None else [],
    )


# ---------------------------------------------------------------------------
# Point-in-Polygon & RestrictedZone Geometry Tests
# ---------------------------------------------------------------------------

def test_point_in_polygon_interior():
    poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((0.5, 0.5), poly) is True


def test_point_in_polygon_exterior():
    poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((1.5, 0.5), poly) is False
    assert point_in_polygon((-0.1, 0.5), poly) is False
    assert point_in_polygon((0.5, -0.1), poly) is False
    assert point_in_polygon((0.5, 1.1), poly) is False


def test_point_in_polygon_boundaries_and_vertices():
    """Boundary edges and vertices are considered inside."""
    poly = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    # Vertices
    assert point_in_polygon((0.2, 0.2), poly) is True
    assert point_in_polygon((0.8, 0.8), poly) is True
    # Edges
    assert point_in_polygon((0.5, 0.2), poly) is True
    assert point_in_polygon((0.8, 0.5), poly) is True
    assert point_in_polygon((0.5, 0.8), poly) is True
    assert point_in_polygon((0.2, 0.5), poly) is True


def test_zone_validation_fewer_than_3_vertices_rejected():
    with pytest.raises(EventConfigurationError, match="at least 3 vertices"):
        RestrictedZone(
            zone_id="z1",
            name="Zone",
            camera_id="CAM_01",
            polygon=((0.1, 0.1), (0.2, 0.2)),
        )


def test_zone_validation_out_of_range_coords_rejected():
    with pytest.raises(EventConfigurationError, match="outside normalized range"):
        RestrictedZone(
            zone_id="z1",
            name="Zone",
            camera_id="CAM_01",
            polygon=((0.0, 0.0), (1.5, 0.5), (0.5, 1.0)),
        )


def test_zone_validation_non_finite_coords_rejected():
    with pytest.raises(EventConfigurationError, match="finite"):
        RestrictedZone(
            zone_id="z1",
            name="Zone",
            camera_id="CAM_01",
            polygon=((0.0, 0.0), (float("nan"), 0.5), (0.5, 1.0)),
        )


def test_zone_validation_empty_id_rejected():
    with pytest.raises(EventConfigurationError, match="zone_id"):
        RestrictedZone(
            zone_id="",
            name="Zone",
            camera_id="CAM_01",
            polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        )


# ---------------------------------------------------------------------------
# SafetyEvent Model Validation Tests
# ---------------------------------------------------------------------------

def test_safety_event_valid():
    ev = SafetyEvent(
        event_id="evt-001",
        event_type="restricted_area_entry",
        camera_id="CAM_01",
        zone_id="zone_01",
        zone_name="Restricted Area",
        track_id=1,
        frame_index=5,
        timestamp_seconds=2.5,
        severity="medium",
        details={"test": 123},
    )
    assert ev.event_id == "evt-001"
    assert ev.event_type == "restricted_area_entry"


def test_safety_event_invalid_type_raises():
    with pytest.raises(ValueError, match="event_type"):
        SafetyEvent(
            event_id="evt-001",
            event_type="unsupported_type",
            camera_id="CAM_01",
            zone_id="zone_01",
            zone_name="Restricted Area",
            track_id=1,
            frame_index=0,
            timestamp_seconds=0.0,
            severity="low",
        )


# ---------------------------------------------------------------------------
# EventEngine Tests: Entry Rule
# ---------------------------------------------------------------------------

def test_entry_emitted_on_observed_outside_to_inside_transition():
    """A person observed outside and then moving inside triggers restricted_area_entry."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=10.0)

    # Frame 1: Person outside (bottom-center at (150, 200) -> norm (0.15, 0.20))
    p_outside = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    events_1 = engine.process_frame(
        _make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_outside]),
        frame_width=1000,
        frame_height=1000,
    )
    assert len(events_1) == 0

    # Frame 2: Person moves inside (bottom-center at (700, 800) -> norm (0.7, 0.8))
    p_inside = _make_tracked_obj(track_id=1, x1=650.0, y1=700.0, x2=750.0, y2=800.0)
    events_2 = engine.process_frame(
        _make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[p_inside]),
        frame_width=1000,
        frame_height=1000,
    )
    assert len(events_2) == 1
    assert events_2[0].event_type == "restricted_area_entry"
    assert events_2[0].track_id == 1
    assert events_2[0].zone_id == "zone_01"


def test_no_false_entry_claim_when_track_first_appears_inside():
    """If a person is first observed already inside the zone, entry must NOT be emitted."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=10.0)

    # First frame: Track appears directly inside (bottom-center at (700, 800) -> norm (0.7, 0.8))
    p_inside = _make_tracked_obj(track_id=1, x1=650.0, y1=700.0, x2=750.0, y2=800.0)
    events = engine.process_frame(
        _make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_inside]),
        frame_width=1000,
        frame_height=1000,
    )
    assert len(events) == 0


def test_no_repeated_entry_events_while_person_stays_inside():
    """A person remaining inside does not emit entry events on subsequent frames."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=10.0)

    # Frame 0: Outside
    p_outside = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_outside]), 1000, 1000)

    # Frame 1: Moves inside -> 1 entry event
    p_inside_1 = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    events_1 = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[p_inside_1]), 1000, 1000)
    assert len(events_1) == 1
    assert events_1[0].event_type == "restricted_area_entry"

    # Frame 2: Still inside -> 0 events
    p_inside_2 = _make_tracked_obj(track_id=1, x1=610.0, y1=610.0, x2=710.0, y2=710.0)
    events_2 = engine.process_frame(_make_frame(frame_index=2, timestamp_seconds=2.0, tracked_objects=[p_inside_2]), 1000, 1000)
    assert len(events_2) == 0


def test_exit_and_reentry_emits_new_entry_event():
    """A person who exits and returns emits a new entry event."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=10.0)

    p_out = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # Frame 0: Outside
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_out]), 1000, 1000)
    # Frame 1: Enters -> 1 entry event
    ev1 = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev1) == 1 and ev1[0].event_type == "restricted_area_entry"

    # Frame 2: Exits
    ev2 = engine.process_frame(_make_frame(frame_index=2, timestamp_seconds=2.0, tracked_objects=[p_out]), 1000, 1000)
    assert len(ev2) == 0

    # Frame 3: Re-enters -> 1 new entry event
    ev3 = engine.process_frame(_make_frame(frame_index=3, timestamp_seconds=3.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev3) == 1
    assert ev3[0].event_type == "restricted_area_entry"


# ---------------------------------------------------------------------------
# EventEngine Tests: Dwell Time Rule
# ---------------------------------------------------------------------------

def test_dwell_event_emitted_at_threshold():
    """Dwell event is emitted when a person remains inside for >= threshold seconds."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=5.0, absence_grace_seconds=1.0)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # Advance frame by frame every 1.0s (within 1.0s grace)
    # t = 0.0s
    assert len(engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t = 1.0s
    assert len(engine.process_frame(_make_frame(frame_index=10, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t = 2.0s
    assert len(engine.process_frame(_make_frame(frame_index=20, timestamp_seconds=2.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t = 3.0s
    assert len(engine.process_frame(_make_frame(frame_index=30, timestamp_seconds=3.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t = 4.0s
    assert len(engine.process_frame(_make_frame(frame_index=40, timestamp_seconds=4.0, tracked_objects=[p_in]), 1000, 1000)) == 0

    # t = 5.0s: threshold reached!
    ev_5 = engine.process_frame(_make_frame(frame_index=50, timestamp_seconds=5.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev_5) == 1
    assert ev_5[0].event_type == "dwell_time_exceeded"
    assert ev_5[0].track_id == 1
    assert ev_5[0].details["dwell_duration_seconds"] == pytest.approx(5.0)


def test_no_duplicate_dwell_event_in_same_continuous_episode():
    """Only one dwell event is emitted during a single continuous presence episode."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=3.0, absence_grace_seconds=1.0)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # Feed frames at t=0, 1, 2, 3
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)
    engine.process_frame(_make_frame(frame_index=10, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)
    engine.process_frame(_make_frame(frame_index=20, timestamp_seconds=2.0, tracked_objects=[p_in]), 1000, 1000)
    ev_3 = engine.process_frame(_make_frame(frame_index=30, timestamp_seconds=3.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev_3) == 1
    assert ev_3[0].event_type == "dwell_time_exceeded"

    # t = 4.0s, 5.0s -> person still inside continuously -> NO duplicate dwell events
    assert len(engine.process_frame(_make_frame(frame_index=40, timestamp_seconds=4.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    assert len(engine.process_frame(_make_frame(frame_index=50, timestamp_seconds=5.0, tracked_objects=[p_in]), 1000, 1000)) == 0


def test_reentry_starts_new_dwell_episode():
    """After leaving and re-entering, a new dwell event can trigger after the threshold."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=2.0, absence_grace_seconds=1.0)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    p_out = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)

    # Episode 1: t=0, 1, 2
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)
    engine.process_frame(_make_frame(frame_index=10, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)
    ev_2 = engine.process_frame(_make_frame(frame_index=20, timestamp_seconds=2.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev_2) == 1 and ev_2[0].event_type == "dwell_time_exceeded"

    # Exit at t=3.0
    engine.process_frame(_make_frame(frame_index=30, timestamp_seconds=3.0, tracked_objects=[p_out]), 1000, 1000)

    # Re-enter at t=4.0 -> entry event
    ev_4 = engine.process_frame(_make_frame(frame_index=40, timestamp_seconds=4.0, tracked_objects=[p_in]), 1000, 1000)
    assert any(e.event_type == "restricted_area_entry" for e in ev_4)

    # Episode 2: t=5.0 (dwell 1.0s), t=6.0 (dwell 2.0s -> threshold reached)
    engine.process_frame(_make_frame(frame_index=50, timestamp_seconds=5.0, tracked_objects=[p_in]), 1000, 1000)
    ev_6 = engine.process_frame(_make_frame(frame_index=60, timestamp_seconds=6.0, tracked_objects=[p_in]), 1000, 1000)
    assert any(e.event_type == "dwell_time_exceeded" for e in ev_6)


# ---------------------------------------------------------------------------
# EventEngine Tests: Tracking Gaps & Grace Period
# ---------------------------------------------------------------------------

def test_short_tracking_gap_tolerated_within_grace():
    """A gap <= absence_grace_seconds does not break continuous presence."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=4.0, absence_grace_seconds=1.5)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # t=0.0: Seen inside
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)
    # t=1.0: Seen inside
    engine.process_frame(_make_frame(frame_index=10, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)

    # t=2.0: Missing in frame (gap = 1.0s <= 1.5s grace)
    engine.process_frame(_make_frame(frame_index=20, timestamp_seconds=2.0, tracked_objects=[]), 1000, 1000)

    # t=3.0: Seen inside again (gap from t=1.0 was 1.0s at t=2.0, now at t=3.0 gap from t=1.0 is 2.0s?
    # Wait, at t=2.0 state was not expired. When t=3.0 is evaluated, time_since_last_seen = 3.0 - 1.0 = 2.0s > 1.5s!
    # To test gap <= 1.5s:
    # If last seen at t=1.0, then at t=2.2 (gap 1.2s <= 1.5s), reappears inside!
    # Let's do that:
    engine2 = EventEngine(zones=[zone], dwell_threshold_seconds=4.0, absence_grace_seconds=1.5)
    # t=0.0
    engine2.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)
    # t=1.0
    engine2.process_frame(_make_frame(frame_index=10, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)
    # t=1.8: frame missing track (gap = 0.8s <= 1.5s)
    engine2.process_frame(_make_frame(frame_index=18, timestamp_seconds=1.8, tracked_objects=[]), 1000, 1000)
    # t=2.2: track returns (gap from last seen t=1.0 is 1.2s <= 1.5s)
    engine2.process_frame(_make_frame(frame_index=22, timestamp_seconds=2.2, tracked_objects=[p_in]), 1000, 1000)
    # t=3.0
    engine2.process_frame(_make_frame(frame_index=30, timestamp_seconds=3.0, tracked_objects=[p_in]), 1000, 1000)
    # t=4.0: dwell reaches 4.0s from t=0.0!
    ev = engine2.process_frame(_make_frame(frame_index=40, timestamp_seconds=4.0, tracked_objects=[p_in]), 1000, 1000)

    assert len(ev) == 1
    assert ev[0].event_type == "dwell_time_exceeded"


def test_long_tracking_gap_resets_presence_and_prevents_false_entry():
    """A gap > absence_grace_seconds resets presence state; reappearing inside starts new dwell without entry."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=3.0, absence_grace_seconds=1.0)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # t=0.0: Seen inside
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)

    # Gap of 3.0s (missing at t=3.0s > 1.0s grace)
    engine.process_frame(_make_frame(frame_index=30, timestamp_seconds=3.0, tracked_objects=[]), 1000, 1000)

    # Reappears inside at t=4.0s:
    # 1. No entry event should be emitted (was not observed outside)
    ev_4 = engine.process_frame(_make_frame(frame_index=40, timestamp_seconds=4.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev_4) == 0

    # 2. Dwell should be reset to count from t=4.0s (not t=0.0s)
    # t=5.0 (elapsed from t=4.0 is 1.0s < 3.0s) -> no dwell event
    assert len(engine.process_frame(_make_frame(frame_index=50, timestamp_seconds=5.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t=6.0 (elapsed 2.0s < 3.0s)
    assert len(engine.process_frame(_make_frame(frame_index=60, timestamp_seconds=6.0, tracked_objects=[p_in]), 1000, 1000)) == 0
    # t=7.0 (elapsed 3.0s == 3.0s) -> dwell event triggered!
    ev_7 = engine.process_frame(_make_frame(frame_index=70, timestamp_seconds=7.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev_7) == 1
    assert ev_7[0].event_type == "dwell_time_exceeded"


# ---------------------------------------------------------------------------
# EventEngine Tests: Camera, Class Filtering & Multi-Track Isolation
# ---------------------------------------------------------------------------

def test_camera_isolation_zones_only_evaluated_for_matching_camera():
    """Zone configured for CAM_01 does not evaluate frames from CAM_02."""
    zone = _make_zone(camera_id="CAM_01")
    engine = EventEngine(zones=[zone])

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    # CAM_02 frame
    events = engine.process_frame(_make_frame(camera_id="CAM_02", tracked_objects=[p_in]), 1000, 1000)
    assert len(events) == 0


def test_person_class_filtering_non_person_ignored():
    """A vehicle (e.g. class_id=2) entering the zone does not trigger person rules."""
    zone = _make_zone()
    engine = EventEngine(zones=[zone], person_class_id=0)

    # Car (class_id=2) starts outside, moves inside
    car_out = _make_tracked_obj(track_id=2, class_id=2, class_name="car", x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    car_in = _make_tracked_obj(track_id=2, class_id=2, class_name="car", x1=600.0, y1=600.0, x2=700.0, y2=700.0)

    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[car_out]), 1000, 1000)
    events = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[car_in]), 1000, 1000)
    assert len(events) == 0


def test_multiple_tracks_tracked_independently():
    """Track 1 and Track 2 have distinct presence and event states."""
    zone = _make_zone(polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=5.0)

    # Frame 0: Track 1 outside, Track 2 inside
    t1_out = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    t2_in = _make_tracked_obj(track_id=2, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    events_0 = engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[t1_out, t2_in]), 1000, 1000)
    assert len(events_0) == 0  # t2 was first seen inside; t1 is outside

    # Frame 1: Track 1 enters, Track 2 stays inside
    t1_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    events_1 = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[t1_in, t2_in]), 1000, 1000)
    assert len(events_1) == 1
    assert events_1[0].event_type == "restricted_area_entry"
    assert events_1[0].track_id == 1

    # Frames 2, 3, 4: both continue presence
    engine.process_frame(_make_frame(frame_index=2, timestamp_seconds=2.0, tracked_objects=[t1_in, t2_in]), 1000, 1000)
    engine.process_frame(_make_frame(frame_index=3, timestamp_seconds=3.0, tracked_objects=[t1_in, t2_in]), 1000, 1000)
    engine.process_frame(_make_frame(frame_index=4, timestamp_seconds=4.0, tracked_objects=[t1_in, t2_in]), 1000, 1000)

    # Frame 5: Track 2 reaches 5.0s dwell (t=0 to t=5); Track 1 is only at 4.0s (t=1 to t=5)
    events_5 = engine.process_frame(_make_frame(frame_index=5, timestamp_seconds=5.0, tracked_objects=[t1_in, t2_in]), 1000, 1000)
    assert len(events_5) == 1
    assert events_5[0].event_type == "dwell_time_exceeded"
    assert events_5[0].track_id == 2


def test_multiple_zones_evaluated_independently():
    """A person can be evaluated against two different zones on the same camera."""
    zone_a = _make_zone(zone_id="zone_a", name="Zone A", polygon=[(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)])
    zone_b = _make_zone(zone_id="zone_b", name="Zone B", polygon=[(0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.6, 1.0)])
    engine = EventEngine(zones=[zone_a, zone_b])

    # Person in middle (outside both): bottom-center (0.5, 0.5)
    p_mid = _make_tracked_obj(track_id=1, x1=450.0, y1=450.0, x2=550.0, y2=500.0)
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_mid]), 1000, 1000)

    # Person moves into Zone A
    p_za = _make_tracked_obj(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    ev_a = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[p_za]), 1000, 1000)
    assert len(ev_a) == 1
    assert ev_a[0].zone_id == "zone_a"

    # Person moves into Zone B
    p_zb = _make_tracked_obj(track_id=1, x1=700.0, y1=700.0, x2=800.0, y2=800.0)
    ev_b = engine.process_frame(_make_frame(frame_index=2, timestamp_seconds=2.0, tracked_objects=[p_zb]), 1000, 1000)
    assert len(ev_b) == 1
    assert ev_b[0].zone_id == "zone_b"


def test_engine_reset_clears_state():
    """Resetting the engine clears dwell tracking and outside history."""
    zone = _make_zone()
    engine = EventEngine(zones=[zone], dwell_threshold_seconds=5.0)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)

    engine.reset()

    # Re-starting at t=0 behaves like a fresh engine
    ev = engine.process_frame(_make_frame(frame_index=0, timestamp_seconds=0.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev) == 0


def test_disabled_engine_emits_no_events():
    """When enabled=False, process_frame immediately returns empty list."""
    zone = _make_zone()
    engine = EventEngine(zones=[zone], enabled=False)

    p_in = _make_tracked_obj(track_id=1, x1=600.0, y1=600.0, x2=700.0, y2=700.0)
    ev = engine.process_frame(_make_frame(frame_index=1, timestamp_seconds=1.0, tracked_objects=[p_in]), 1000, 1000)
    assert len(ev) == 0


# ---------------------------------------------------------------------------
# CLI Tests (no model downloads, network, or real video)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from aegis_ad.detection.models import DetectionResult


def _make_mock_event_settings(event_cfg: dict | None = None) -> MagicMock:
    mock = MagicMock()
    mock.raw = {
        "detection": {"model_path": "mock.pt", "confidence_threshold": 0.25, "device": "cpu"},
        "tracking": {"track_buffer": 30},
        "events": event_cfg or {
            "enabled": True,
            "person_class_id": 0,
            "dwell_threshold_seconds": 30.0,
            "absence_grace_seconds": 1.0,
            "zones": [
                {
                    "zone_id": "zone_01",
                    "name": "Restricted Area",
                    "camera_id": "CAM_01",
                    "polygon": [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]],
                }
            ],
        },
    }
    return mock


def _run_event_cli(args: list[str], event_cfg: dict | None = None) -> int:
    from aegis_ad.events.__main__ import run_cli

    mock_video = MagicMock()
    mock_video.metadata.camera_id = "CAM_01"
    mock_video.metadata.source_path = "mock.mp4"
    mock_video.metadata.width = 640
    mock_video.metadata.height = 480
    mock_video.metadata.fps = 25.0
    mock_video.read_frames.return_value = iter([])

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        camera_id="CAM_01", frame_index=0, timestamp_seconds=0.0,
        detections=[], inference_time_ms=1.0,
    )

    mock_tracker = MagicMock()
    mock_tracker.update.return_value = TrackedFrame(
        camera_id="CAM_01", frame_index=0, timestamp_seconds=0.0,
        tracked_objects=[],
    )

    with (
        patch("aegis_ad.events.__main__.load_settings", return_value=_make_mock_event_settings(event_cfg)),
        patch("aegis_ad.events.__main__.VideoSource", return_value=mock_video),
        patch("aegis_ad.events.__main__.YOLODetector", return_value=mock_detector),
        patch("aegis_ad.events.__main__.ObjectTracker", return_value=mock_tracker),
        patch("pathlib.Path.exists", return_value=True),
    ):
        return run_cli(args)


def test_event_cli_runs_successfully():
    """Event CLI returns 0 on a mocked video pipeline."""
    code = _run_event_cli(["--source", "any.mp4", "--camera-id", "CAM_01"])
    assert code == 0


def test_event_cli_missing_source_returns_1():
    """Event CLI returns 1 if video source does not exist."""
    from aegis_ad.events.__main__ import run_cli
    with patch("pathlib.Path.exists", return_value=False):
        code = run_cli(["--source", "nonexistent.mp4"])
    assert code == 1


def test_event_cli_dwell_override_applied():
    """Command-line --dwell overrides dwell_threshold_seconds in event engine."""
    from aegis_ad.events.__main__ import run_cli

    engines_created: list[EventEngine] = []
    original_from_config = EventEngine.from_config

    def _spy_from_config(cfg):
        eng = original_from_config(cfg)
        engines_created.append(eng)
        return eng

    mock_video = MagicMock()
    mock_video.metadata.camera_id = "CAM_01"
    mock_video.metadata.source_path = "mock.mp4"
    mock_video.metadata.width = 640
    mock_video.metadata.height = 480
    mock_video.metadata.fps = 25.0
    mock_video.read_frames.return_value = iter([])

    with (
        patch("aegis_ad.events.__main__.load_settings", return_value=_make_mock_event_settings()),
        patch("aegis_ad.events.__main__.VideoSource", return_value=mock_video),
        patch("aegis_ad.events.__main__.YOLODetector"),
        patch("aegis_ad.events.__main__.ObjectTracker"),
        patch.object(EventEngine, "from_config", side_effect=_spy_from_config),
        patch("pathlib.Path.exists", return_value=True),
    ):
        code = run_cli(["--source", "any.mp4", "--dwell", "12.5"])

    assert code == 0
    assert len(engines_created) == 1
    assert engines_created[0].dwell_threshold_seconds == pytest.approx(12.5)
