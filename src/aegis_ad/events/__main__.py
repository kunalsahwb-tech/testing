"""CLI smoke test for the AEGIS-AD safety event pipeline.

Connects: VideoSource -> YOLODetector -> ObjectTracker -> EventEngine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aegis_ad.alerts.models import Alert
from aegis_ad.alerts.repository import AlertRepository
from aegis_ad.alerts.service import AlertService
from aegis_ad.detection.detector import YOLODetector
from aegis_ad.detection.models import DetectionError
from aegis_ad.evidence.store import EvidenceStore
from aegis_ad.events.engine import EventEngine
from aegis_ad.events.models import EventConfigurationError, SafetyEvent
from aegis_ad.ingestion.models import IngestionError
from aegis_ad.ingestion.video_source import VideoSource
from aegis_ad.settings import ConfigurationError, load_settings
from aegis_ad.tracking.models import TrackingError
from aegis_ad.tracking.tracker import ObjectTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegis_ad.events",
        description="AEGIS-AD Safety Event Pipeline Smoke Test",
    )
    parser.add_argument("--source", "-s", type=str, default="data/videos/sample.mp4",
                        help="Path to the input video file.")
    parser.add_argument("--camera-id", "-c", type=str, default="CAM_01",
                        help="Camera identifier.")
    parser.add_argument("--max-frames", "-n", type=int, default=10,
                        help="Maximum frames to process (<=0 for all).")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO model path or name (overrides config).")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (overrides config).")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: cpu or cuda:N (overrides config).")
    parser.add_argument("--dwell", type=float, default=None,
                        help="Override dwell threshold in seconds.")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to alerts SQLite database (overrides config).")
    parser.add_argument("--evidence-dir", type=str, default=None,
                        help="Directory for saving evidence snapshots (overrides config).")
    return parser


def run_cli(args: list[str] | None = None) -> int:
    """Run the safety event pipeline smoke test."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    source_path = Path(parsed.source)
    if not source_path.exists():
        print(f"[ERROR] Video file not found: {source_path}", file=sys.stderr)
        print("Supply an MP4 with --source path/to/video.mp4", file=sys.stderr)
        return 1

    # Load configuration
    try:
        settings = load_settings()
        det_cfg = settings.raw.get("detection", {})
        trk_cfg = settings.raw.get("tracking", {})
        evt_cfg = settings.raw.get("events", {})
        alerts_cfg = settings.raw.get("alerts", {})
        evidence_cfg = settings.raw.get("evidence", {})
    except ConfigurationError as exc:
        print(f"[ERROR] Configuration failed: {exc}", file=sys.stderr)
        return 1

    # Detection parameters
    model_path = parsed.model or det_cfg.get("model_path", "yolo26n.pt")
    confidence = parsed.conf or det_cfg.get("confidence_threshold", 0.25)
    device = parsed.device or det_cfg.get("device", "cpu")
    image_size: int = det_cfg.get("image_size", 640)
    raw_targets = det_cfg.get("target_classes")
    target_class_ids = frozenset(raw_targets) if raw_targets is not None else None

    # Tracking parameters
    track_high_thresh: float = trk_cfg.get("track_high_thresh", 0.5)
    track_low_thresh: float = trk_cfg.get("track_low_thresh", 0.1)
    new_track_thresh: float = trk_cfg.get("new_track_thresh", 0.6)
    track_buffer: int = trk_cfg.get("track_buffer", 30)
    match_thresh: float = trk_cfg.get("match_thresh", 0.8)

    # Event parameters
    dwell_override = parsed.dwell
    if dwell_override is not None:
        evt_cfg = dict(evt_cfg)
        evt_cfg["dwell_threshold_seconds"] = dwell_override

    # Alert and evidence parameters
    db_path = parsed.db_path or alerts_cfg.get("db_path", "data/alerts.db")
    evidence_dir = parsed.evidence_dir or evidence_cfg.get(
        "evidence_dir",
        settings.raw.get("paths", {}).get("evidence_dir", "data/evidence"),
    )

    max_frames = parsed.max_frames if parsed.max_frames > 0 else None
    camera_id = parsed.camera_id

    print("=" * 56)
    print(" AEGIS-AD Safety Event Pipeline Smoke Test")
    print("=" * 56)
    print(f"Model:        {model_path} ({device})")
    print(f"Confidence:   {confidence}")
    print(f"Alerts DB:    {db_path}")
    print(f"Evidence Dir: {evidence_dir}")

    # Initialise alert repository and evidence store
    try:
        alert_repo = AlertRepository(db_path=Path(db_path))
        evidence_store = EvidenceStore(evidence_dir=Path(evidence_dir))
        alert_service = AlertService(repository=alert_repo, evidence_store=evidence_store)
    except Exception as exc:
        print(f"[ERROR] Alert service initialisation failed: {exc}", file=sys.stderr)
        return 1

    # Initialise video source
    try:
        video = VideoSource(source_path, camera_id=camera_id)
    except (FileNotFoundError, IngestionError, ValueError) as exc:
        print(f"[ERROR] Video source failed: {exc}", file=sys.stderr)
        return 1

    meta = video.metadata
    print(f"Camera ID:   {meta.camera_id}")
    print(f"Source:      {meta.source_path}")
    print(f"Resolution:  {meta.width}x{meta.height}  FPS: {meta.fps:.2f}")
    print(f"Limit:       {max_frames if max_frames else 'All'} frames")

    # Initialise Event Engine
    try:
        event_engine = EventEngine.from_config(evt_cfg)
    except (EventConfigurationError, ValueError) as exc:
        print(f"[ERROR] Event engine configuration failed: {exc}", file=sys.stderr)
        return 1

    active_zones = [z for z in event_engine.zones if z.camera_id == camera_id]
    print(f"Event Rules: {'ENABLED' if event_engine.enabled else 'DISABLED'}")
    print(f"Active Zones:{len(active_zones)} configured for {camera_id}")
    for z in active_zones:
        print(f"  - [{z.zone_id}] {z.name} ({len(z.polygon)} vertices)")
    print(f"Dwell Thresh:{event_engine.dwell_threshold_seconds:.1f}s (grace: {event_engine.absence_grace_seconds:.1f}s)")
    print("-" * 56)

    # Initialise YOLO detector
    print(f"Loading model '{model_path}' — this may download weights on first run...")
    try:
        detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence,
            image_size=image_size,
            device=device,
            target_class_ids=target_class_ids,
        )
    except DetectionError as exc:
        print(f"[ERROR] Model load failed: {exc}", file=sys.stderr)
        return 1
    print("Model ready.")

    # Initialise tracker
    try:
        tracker = ObjectTracker(
            camera_id=camera_id,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
    except (ValueError, TrackingError) as exc:
        print(f"[ERROR] Tracker initialisation failed: {exc}", file=sys.stderr)
        return 1
    print("Tracker ready.")
    print("-" * 56)

    total_frames = 0
    total_detections = 0
    total_tracked = 0
    all_events: list[SafetyEvent] = []
    all_alerts: list[Alert] = []
    evidence_captured = 0
    evidence_failed = 0
    evidence_none = 0

    try:
        for packet in video.read_frames(max_frames=max_frames):
            # 1. Detection
            try:
                det_result = detector.detect(packet)
            except DetectionError as exc:
                print(f"[WARN] Detection error: {exc}", file=sys.stderr)
                continue

            # 2. Tracking
            try:
                trk_frame = tracker.update(det_result)
            except TrackingError as exc:
                print(f"[WARN] Tracking error: {exc}", file=sys.stderr)
                continue

            # 3. Event rules
            new_events = event_engine.process_frame(
                frame=trk_frame,
                frame_width=meta.width,
                frame_height=meta.height,
            )

            total_frames += 1
            total_detections += det_result.detection_count
            total_tracked += trk_frame.tracked_count
            all_events.extend(new_events)

            if not new_events:
                print(
                    f"[{trk_frame.camera_id}] Frame {trk_frame.frame_index:04d} | "
                    f"Time: {trk_frame.timestamp_seconds:6.3f}s | "
                    f"det={det_result.detection_count} | tracks={trk_frame.tracked_count} | events=0"
                )
            else:
                for evt in new_events:
                    alert = alert_service.process_event(event=evt, frame=packet.frame)
                    all_alerts.append(alert)
                    if alert.evidence_status == "captured":
                        evidence_captured += 1
                    elif alert.evidence_status == "failed":
                        evidence_failed += 1
                    else:
                        evidence_none += 1

                    ev_info = f"evidence={alert.evidence_status}"
                    if alert.evidence_path:
                        ev_info += f" ({alert.evidence_path})"

                    print(
                        f"[{evt.camera_id}] Frame {evt.frame_index:04d} | "
                        f"Time: {evt.timestamp_seconds:6.3f}s | "
                        f"[ALERT: {evt.event_type.upper()}] track={evt.track_id} "
                        f"zone='{evt.zone_name}' sev={evt.severity} | {ev_info}"
                    )
    except Exception as exc:
        print(f"[ERROR] Unexpected failure during processing: {exc}", file=sys.stderr)
        return 1

    print("-" * 56)
    print(f"Summary: Processed {total_frames} frames.")
    print(f"  Detections total:     {total_detections}")
    print(f"  Tracked objects seen: {total_tracked}")
    print(f"  Safety events total:  {len(all_events)}")
    if all_events:
        events_by_type: dict[str, int] = {}
        for ev in all_events:
            events_by_type[ev.event_type] = events_by_type.get(ev.event_type, 0) + 1
        for etype, count in events_by_type.items():
            print(f"    - {etype}: {count}")
    print(f"  Alerts generated:     {len(all_alerts)}")
    print(f"  Evidence captured:    {evidence_captured}")
    if evidence_failed > 0:
        print(f"  Evidence failures:    {evidence_failed}")
    print(f"  Alerts database:      {db_path}")
    if total_frames > 0 and total_detections == 0:
        print("  Note: Zero detections on synthetic sample video is expected.")
    print("=" * 56)
    return 0


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
