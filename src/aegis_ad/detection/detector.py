"""YOLO-based object detector for AEGIS-AD."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from aegis_ad.detection.models import BoundingBox, Detection, DetectionError, DetectionResult
from aegis_ad.ingestion.models import FramePacket

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# COCO class IDs for the target safety-relevant classes
TARGET_CLASS_IDS: frozenset[int] = frozenset({
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
})

COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLODetector:
    """Wraps an Ultralytics YOLO model to detect persons and vehicles in video frames.

    The model is loaded at construction time and reused across frames.
    Model weights are not downloaded during import — only when a YOLODetector is instantiated.
    """

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: str = "cpu",
        target_class_ids: frozenset[int] | None = None,
    ) -> None:
        """Initialise the detector and load the YOLO model.

        Args:
            model_path: Path or name of the pretrained YOLO model file.
            confidence_threshold: Minimum confidence score for a detection to be included.
            image_size: Input image size used for inference.
            device: Compute device ("cpu" or "cuda:N").
            target_class_ids: Set of COCO class IDs to retain. Defaults to TARGET_CLASS_IDS.
        """
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in (0.0, 1.0], got: {confidence_threshold}"
            )
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got: {image_size}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.target_class_ids: frozenset[int] = (
            target_class_ids if target_class_ids is not None else TARGET_CLASS_IDS
        )

        logger.info("Loading YOLO model from '%s' on device '%s'", model_path, device)
        try:
            # Import Ultralytics only here so the package import does not trigger a download
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = YOLO(model_path)
            self._model.to(device)
        except Exception as exc:
            raise DetectionError(
                f"Failed to load YOLO model from '{model_path}': {exc}"
            ) from exc

        logger.info("Model loaded successfully: %s", model_path)

    def detect(self, packet: FramePacket) -> DetectionResult:
        """Run inference on a FramePacket and return a DetectionResult.

        Args:
            packet: A FramePacket from VideoSource.

        Returns:
            DetectionResult preserving camera_id, frame_index, and timestamp.

        Raises:
            DetectionError: If the frame is invalid or inference fails.
        """
        if not isinstance(packet.frame, np.ndarray) or packet.frame.size == 0:
            raise DetectionError(
                f"[{packet.camera_id}] Frame {packet.frame_index}: invalid or empty frame array."
            )

        t_start = time.perf_counter()
        try:
            results = self._model(
                packet.frame,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectionError(
                f"[{packet.camera_id}] Frame {packet.frame_index}: inference failed: {exc}"
            ) from exc

        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        detections = self._parse_results(results)

        return DetectionResult(
            camera_id=packet.camera_id,
            frame_index=packet.frame_index,
            timestamp_seconds=packet.timestamp_seconds,
            detections=detections,
            inference_time_ms=inference_time_ms,
        )

    def _parse_results(self, results: list[Any]) -> list[Detection]:
        """Convert raw Ultralytics results into Detection objects."""
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id not in self.target_class_ids:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

            # Guard against degenerate boxes from model output
            if x2 <= x1 or y2 <= y1:
                logger.debug("Skipping degenerate box: [%.1f,%.1f,%.1f,%.1f]", x1, y1, x2, y2)
                continue

            class_name = COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return detections
