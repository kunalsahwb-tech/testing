"""Object detection module for AEGIS-AD."""

from aegis_ad.detection.models import BoundingBox, Detection, DetectionError, DetectionResult
from aegis_ad.detection.detector import YOLODetector, TARGET_CLASS_IDS, COCO_CLASS_NAMES

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionError",
    "DetectionResult",
    "YOLODetector",
    "TARGET_CLASS_IDS",
    "COCO_CLASS_NAMES",
]
