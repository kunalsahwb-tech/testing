"""Video source implementation for reading recorded video files."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Generator, Iterator

import cv2

from aegis_ad.ingestion.models import FramePacket, IngestionError, VideoMetadata

logger = logging.getLogger(__name__)


class VideoSource:
    """Reads recorded video files frame-by-frame and yields FramePackets."""

    def __init__(self, file_path: str | Path, camera_id: str) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

        self.file_path = Path(file_path).resolve()
        self.camera_id = camera_id

        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise IngestionError(f"Path is not a regular file: {self.file_path}")

        self._open()

    def _open(self) -> None:
        """Initialize OpenCV capture and extract video metadata."""
        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            self.close()
            raise IngestionError(f"OpenCV failed to open video file: {self.file_path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        total_frames_prop = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_prop if total_frames_prop > 0 else None

        if width <= 0 or height <= 0:
            self.close()
            raise IngestionError(
                f"Invalid video dimensions ({width}x{height}) in {self.file_path}"
            )

        if not math.isfinite(fps) or fps <= 0.0 or fps > 1000.0:
            self.close()
            raise IngestionError(
                f"Invalid or undetectable FPS ({fps}) in {self.file_path}"
            )

        self._metadata = VideoMetadata(
            source_path=str(self.file_path),
            camera_id=self.camera_id,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
        )
        logger.debug(
            "Opened video %s: %dx%d @ %.2f fps, total_frames=%s",
            self.file_path.name,
            width,
            height,
            fps,
            total_frames,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Return the video metadata."""
        if self._metadata is None:
            raise IngestionError("Video source is closed or uninitialized.")
        return self._metadata

    @property
    def is_opened(self) -> bool:
        """Return True if the video capture is active."""
        return self._cap is not None and self._cap.isOpened()

    def read_frames(self, max_frames: int | None = None) -> Generator[FramePacket, None, None]:
        """Yield FramePacket objects sequentially.

        Args:
            max_frames: Optional upper limit on the number of frames to yield.
        """
        if not self.is_opened or self._cap is None:
            raise IngestionError("Cannot read frames from a closed or uninitialized video source.")

        fps = self.metadata.fps
        frame_index = 0

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    # End of stream or unreadable frame
                    break

                timestamp_seconds = frame_index / fps

                yield FramePacket(
                    camera_id=self.camera_id,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    frame=frame,
                    source=str(self.file_path),
                )
                frame_index += 1
        finally:
            self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        return self.read_frames()

    def close(self) -> None:
        """Release OpenCV video capture resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Error releasing VideoCapture: %s", exc)
            finally:
                self._cap = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
