"""CLI entrypoint for object detection smoke testing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from aegis_ad.detection.models import DetectionError
from aegis_ad.detection.detector import YOLODetector
from aegis_ad.ingestion.models import IngestionError
from aegis_ad.ingestion.video_source import VideoSource
from aegis_ad.settings import ConfigurationError, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegis_ad.detection",
        description="AEGIS-AD Object Detection Smoke Test",
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
    """Run the detection smoke test."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    source_path = Path(parsed.source)
    if not source_path.exists():
        print(f"[ERROR] Video file not found: {source_path}", file=sys.stderr)
        print("Supply an MP4 with --source path/to/video.mp4", file=sys.stderr)
        return 1

    # Load configuration for defaults
    try:
        settings = load_settings()
        det_cfg = settings.raw.get("detection", {})
    except ConfigurationError as exc:
        print(f"[ERROR] Configuration failed: {exc}", file=sys.stderr)
        return 1

    model_path = parsed.model or det_cfg.get("model_path", "yolo26n.pt")
    confidence = parsed.conf or det_cfg.get("confidence_threshold", 0.25)
    device = parsed.device or det_cfg.get("device", "cpu")
    image_size: int = det_cfg.get("image_size", 640)

    raw_targets = det_cfg.get("target_classes")
    target_class_ids: frozenset[int] | None = (
        frozenset(raw_targets) if raw_targets is not None else None
    )

    max_frames = parsed.max_frames if parsed.max_frames > 0 else None

    print("=" * 48)
    print(" AEGIS-AD Object Detection Smoke Test")
    print("=" * 48)
    print(f"Model:       {model_path}")
    print(f"Device:      {device}")
    print(f"Confidence:  {confidence}")
    print(f"Image size:  {image_size}")

    # Initialise video source
    try:
        video = VideoSource(source_path, camera_id=parsed.camera_id)
    except (FileNotFoundError, IngestionError, ValueError) as exc:
        print(f"[ERROR] Video source failed: {exc}", file=sys.stderr)
        return 1

    meta = video.metadata
    print(f"Camera ID:   {meta.camera_id}")
    print(f"Source:      {meta.source_path}")
    print(f"Resolution:  {meta.width}x{meta.height}  FPS: {meta.fps:.2f}")
    print(f"Limit:       {max_frames if max_frames else 'All'} frames")
    print("-" * 48)

    # Initialise detector (model download happens here if needed)
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
    print("-" * 48)

    total_frames = 0
    total_detections = 0
    total_inference_ms = 0.0

    try:
        for packet in video.read_frames(max_frames=max_frames):
            try:
                result = detector.detect(packet)
            except DetectionError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
                continue

            total_frames += 1
            total_detections += result.detection_count
            total_inference_ms += result.inference_time_ms

            if result.detection_count == 0:
                print(
                    f"[{result.camera_id}] Frame {result.frame_index:04d} | "
                    f"Time: {result.timestamp_seconds:6.3f}s | "
                    f"No detections above threshold"
                )
            else:
                for det in result.detections:
                    bb = det.bounding_box
                    print(
                        f"[{result.camera_id}] Frame {result.frame_index:04d} | "
                        f"Time: {result.timestamp_seconds:6.3f}s | "
                        f"{det.class_name} conf={det.confidence:.2f} "
                        f"box=[{bb.x1:.0f},{bb.y1:.0f},{bb.x2:.0f},{bb.y2:.0f}]"
                    )
    except Exception as exc:
        print(f"[ERROR] Unexpected failure during processing: {exc}", file=sys.stderr)
        return 1

    avg_ms = total_inference_ms / total_frames if total_frames > 0 else 0.0
    print("-" * 48)
    print(f"Summary: Processed {total_frames} frames, {total_detections} total detections.")
    print(f"Inference time: total={total_inference_ms:.1f}ms, avg={avg_ms:.1f}ms/frame.")
    print("=" * 48)
    return 0


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
