"""Tests for the video ingestion module."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from aegis_ad.ingestion import FramePacket, IngestionError, VideoMetadata, VideoSource


def create_synthetic_video(
    file_path: Path,
    num_frames: int = 15,
    width: int = 64,
    height: int = 48,
    fps: float = 10.0,
) -> Path:
    """Generate a tiny valid MP4 video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to MJPG if mp4v is not supported in the environment
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return file_path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Fixture providing a valid 15-frame test video."""
    video_path = tmp_path / "test_feed.mp4"
    return create_synthetic_video(video_path, num_frames=15, width=64, height=48, fps=10.0)


def test_valid_video_metadata(sample_video: Path):
    """Test 1: A valid video opens and its metadata is correct."""
    source = VideoSource(sample_video, camera_id="CAM_TEST")
    try:
        meta = source.metadata
        assert isinstance(meta, VideoMetadata)
        assert meta.camera_id == "CAM_TEST"
        assert meta.width == 64
        assert meta.height == 48
        assert abs(meta.fps - 10.0) < 0.1
        assert meta.total_frames == 15
        assert meta.source_path == str(sample_video.resolve())
    finally:
        source.close()


def test_frame_packets_order_and_indexing(sample_video: Path):
    """Tests 2, 3, 4, 5: Correct order, 0-based indices, timestamps, and frame count."""
    source = VideoSource(sample_video, camera_id="CAM_01")
    packets = list(source.read_frames())

    assert len(packets) == 15, f"Expected 15 frames, got {len(packets)}"

    for expected_index, packet in enumerate(packets):
        assert isinstance(packet, FramePacket)
        assert packet.camera_id == "CAM_01"
        assert packet.frame_index == expected_index
        expected_timestamp = expected_index / 10.0
        assert abs(packet.timestamp_seconds - expected_timestamp) < 1e-4
        assert packet.frame.shape == (48, 64, 3)
        assert packet.frame.dtype == np.uint8
        assert packet.source == str(sample_video.resolve())


def test_max_frames_limit(sample_video: Path):
    """Test that max_frames limits yielded packets and exits cleanly."""
    source = VideoSource(sample_video, camera_id="CAM_01")
    packets = list(source.read_frames(max_frames=5))
    assert len(packets) == 5
    assert packets[-1].frame_index == 4


def test_missing_video_file_raises_clear_error(tmp_path: Path):
    """Test 6: A missing file produces a clear FileNotFoundError."""
    missing_path = tmp_path / "non_existent.mp4"
    with pytest.raises(FileNotFoundError) as exc_info:
        VideoSource(missing_path, camera_id="CAM_01")
    assert "Video file not found" in str(exc_info.value)


def test_invalid_unreadable_file_raises_clear_error(tmp_path: Path):
    """Test 7: An invalid or corrupted file produces a clear IngestionError."""
    corrupted_file = tmp_path / "corrupt.mp4"
    corrupted_file.write_bytes(b"not a valid video content header")

    with pytest.raises(IngestionError) as exc_info:
        VideoSource(corrupted_file, camera_id="CAM_01")
    assert "OpenCV failed to open" in str(exc_info.value) or "Invalid" in str(exc_info.value)


def test_resource_release_after_full_iteration(sample_video: Path):
    """Test 8a: Video resources are released after iteration completes normally."""
    source = VideoSource(sample_video, camera_id="CAM_01")
    assert source.is_opened
    _ = list(source.read_frames())
    assert not source.is_opened


def test_resource_release_on_early_termination(sample_video: Path):
    """Test 8b: Video resources are released when iteration is interrupted early."""
    source = VideoSource(sample_video, camera_id="CAM_01")
    assert source.is_opened
    for packet in source.read_frames():
        if packet.frame_index == 2:
            break
    assert not source.is_opened


def test_context_manager_releases_resource(sample_video: Path):
    """Test 8c: Using VideoSource as context manager releases resources."""
    with VideoSource(sample_video, camera_id="CAM_01") as source:
        assert source.is_opened
    assert not source.is_opened


def test_empty_camera_id_raises_value_error(sample_video: Path):
    """Test that empty or non-string camera_id raises ValueError."""
    with pytest.raises(ValueError, match="camera_id"):
        VideoSource(sample_video, camera_id="")


def test_cli_smoke_test_success(sample_video: Path, capsys: pytest.CaptureFixture):
    """Test CLI smoke test execution with valid video."""
    from aegis_ad.ingestion.__main__ import run_cli

    exit_code = run_cli(["--source", str(sample_video), "--camera-id", "CAM_CLI", "--max-frames", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AEGIS-AD Video Ingestion Smoke Test" in captured.out
    assert "Camera ID:    CAM_CLI" in captured.out
    assert "Summary: Ingested 3 frames" in captured.out
    assert "Video capture resource released." in captured.out


def test_cli_smoke_test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test CLI smoke test reports clear error and guidance on missing file."""
    from aegis_ad.ingestion.__main__ import run_cli

    missing_video = tmp_path / "absent.mp4"
    exit_code = run_cli(["--source", str(missing_video)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Video file not found" in captured.err


@pytest.mark.parametrize("bad_fps", [0.0, -5.0, float("nan"), float("inf"), -float("inf"), 1500.0])
def test_invalid_or_unavailable_fps_raises_ingestion_error(
    sample_video: Path, monkeypatch: pytest.MonkeyPatch, bad_fps: float
):
    """Verify that zero, negative, NaN, Inf, and out-of-range FPS raise IngestionError."""
    original_get = cv2.VideoCapture.get

    def mock_get(self, propId: int):
        if propId == cv2.CAP_PROP_FPS:
            return bad_fps
        return original_get(self, propId)

    monkeypatch.setattr(cv2.VideoCapture, "get", mock_get)

    with pytest.raises(IngestionError) as exc_info:
        VideoSource(sample_video, camera_id="CAM_01")
    assert "Invalid or undetectable FPS" in str(exc_info.value)


