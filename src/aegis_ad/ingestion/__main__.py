"""CLI entrypoint for video ingestion smoke testing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from aegis_ad.ingestion.models import IngestionError
from aegis_ad.ingestion.video_source import VideoSource


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m aegis_ad.ingestion",
        description="AEGIS-AD Video Ingestion Smoke Test",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="data/videos/sample.mp4",
        help="Path to the input video file (e.g., MP4).",
    )
    parser.add_argument(
        "--camera-id",
        "-c",
        type=str,
        default="CAM_01",
        help="Identifier for the camera feed (default: CAM_01).",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=10,
        help="Maximum number of frames to ingest (default: 10, set <= 0 for all).",
    )
    return parser


def run_cli(args: list[str] | None = None) -> int:
    """Run ingestion smoke test."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    source_path = Path(parsed.source)
    if not source_path.exists():
        print(f"[ERROR] Video file not found: {source_path}", file=sys.stderr)
        print("To run ingestion on a video, specify an existing MP4:", file=sys.stderr)
        print(f"  python -m aegis_ad.ingestion --source path/to/video.mp4 --camera-id {parsed.camera_id}", file=sys.stderr)
        print(f"Or place an MP4 at: data/videos/sample.mp4", file=sys.stderr)
        return 1

    max_frames = parsed.max_frames if parsed.max_frames > 0 else None

    print("========================================")
    print(" AEGIS-AD Video Ingestion Smoke Test")
    print("========================================")

    try:
        source = VideoSource(source_path, camera_id=parsed.camera_id)
    except (FileNotFoundError, IngestionError, ValueError) as exc:
        print(f"[ERROR] Failed to initialize video source: {exc}", file=sys.stderr)
        return 1

    meta = source.metadata
    print(f"Camera ID:    {meta.camera_id}")
    print(f"Source:       {meta.source_path}")
    print(f"Resolution:   {meta.width}x{meta.height}")
    print(f"FPS:          {meta.fps:.2f}")
    print(f"Total Frames: {meta.total_frames if meta.total_frames is not None else 'Unknown'}")
    print(f"Limit:        {max_frames if max_frames is not None else 'All'} frames")
    print("----------------------------------------")

    start_time = time.perf_counter()
    count = 0

    try:
        for packet in source.read_frames(max_frames=max_frames):
            count += 1
            h, w = packet.frame.shape[:2]
            print(
                f"[{packet.camera_id}] Frame {packet.frame_index:04d} | "
                f"Time: {packet.timestamp_seconds:6.3f}s | "
                f"Shape: {w}x{h}"
            )
    except Exception as exc:
        print(f"[ERROR] Ingestion interrupted: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start_time
    fps_processed = (count / elapsed) if elapsed > 0 else 0.0

    print("----------------------------------------")
    print(f"Summary: Ingested {count} frames in {elapsed:.3f}s ({fps_processed:.1f} fps).")
    print("Video capture resource released.")
    print("========================================")
    return 0


def main() -> None:
    """Entry point for python -m aegis_ad.ingestion."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
