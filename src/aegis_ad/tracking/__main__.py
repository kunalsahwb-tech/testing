"""CLI smoke test for the AEGIS-AD tracking pipeline.

Connects: VideoSource -> YOLODetector -> ObjectTracker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aegis_ad.detection.models import DetectionError
from aegis_ad.detection.detector import YOLODetector
from aegis_ad.ingestion.models import IngestionError
from aegis_ad.ingestion.video_source import VideoSource
from aegis_ad.settings import ConfigurationError, load_settings
from aegis_ad.tracking.models import TrackingError
from aegis_ad.tracking.tracker import ObjectTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegis_ad.tracking",
        description="AEGIS-AD Tracking Pipeline Smoke Test",
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
    return parser


def run_cli(args: list[str] | None = None) -> int:
    """Run the tracking pipeline smoke test."""
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
    except ConfigurationError as exc:
        print(f"[ERROR] Configuration failed: {exc}", file=sys.stderr)
        return 1

    model_path = parsed.model or det_cfg.get("model_path", "yolo26n.pt")
    confidence = parsed.conf or det_cfg.get("confidence_threshold", 0.25)
    device = parsed.device or det_cfg.get("device", "cpu")
    image_size: int = det_cfg.get("image_size", 640)
    raw_targets = det_cfg.get("target_classes")
    target_class_ids = frozenset(raw_targets) if raw_targets is not None else None

    track_high_thresh: float = trk_cfg.get("track_high_thresh", 0.5)
    track_low_thresh: float = trk_cfg.get("track_low_thresh", 0.1)
    new_track_thresh: float = trk_cfg.get("new_track_thresh", 0.6)
    track_buffer: int = trk_cfg.get("track_buffer", 30)
    match_thresh: float = trk_cfg.get("match_thresh", 0.8)

    max_frames = parsed.max_frames if parsed.max_frames > 0 else None
    camera_id = parsed.camera_id

    print("=" * 52)
    print(" AEGIS-AD Tracking Pipeline Smoke Test")
    print("=" * 52)
    print(f"Model:       {model_path}")
    print(f"Device:      {device}")
    print(f"Confidence:  {confidence}")
    print(f"Image size:  {image_size}")
    print(f"TrackBuffer: {track_buffer}")

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
    print("-" * 52)

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

    # Initialise per-camera tracker
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
    print("-" * 52)

    total_frames = 0
    total_detections = 0
    total_tracked = 0
    all_track_ids: set[int] = set()

    try:
        for packet in video.read_frames(max_frames=max_frames):
            # Detection
            try:
                det_result = detector.detect(packet)
            except DetectionError as exc:
                print(f"[WARN] Detection failed: {exc}", file=sys.stderr)
                continue

            # Tracking
            try:
                trk_frame = tracker.update(det_result)
            except TrackingError as exc:
                print(f"[WARN] Tracking failed: {exc}", file=sys.stderr)
                continue

            total_frames += 1
            total_detections += det_result.detection_count
            total_tracked += trk_frame.tracked_count
            all_track_ids.update(trk_frame.track_ids)

            if trk_frame.tracked_count == 0:
                print(
                    f"[{trk_frame.camera_id}] Frame {trk_frame.frame_index:04d} | "
                    f"Time: {trk_frame.timestamp_seconds:6.3f}s | "
                    f"det={det_result.detection_count} | tracks=0"
                )
            else:
                for obj in trk_frame.tracked_objects:
                    print(
                        f"[{trk_frame.camera_id}] Frame {trk_frame.frame_index:04d} | "
                        f"Time: {trk_frame.timestamp_seconds:6.3f}s | "
                        f"track_id={obj.track_id} {obj.class_name} "
                        f"conf={obj.confidence:.2f} "
                        f"box=[{obj.x1:.0f},{obj.y1:.0f},{obj.x2:.0f},{obj.y2:.0f}]"
                    )
    except Exception as exc:
        print(f"[ERROR] Unexpected failure during processing: {exc}", file=sys.stderr)
        return 1

    print("-" * 52)
    print(f"Summary: Processed {total_frames} frames.")
    print(f"  Detections total:     {total_detections}")
    print(f"  Tracked objects seen: {total_tracked}")
    print(f"  Unique track IDs:     {len(all_track_ids)}")
    if total_frames > 0 and total_detections == 0:
        print("  Note: Zero detections on synthetic video is expected.")
    print("=" * 52)
    return 0


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
