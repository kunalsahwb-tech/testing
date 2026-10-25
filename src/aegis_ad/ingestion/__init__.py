"""Video ingestion module for AEGIS-AD."""

from aegis_ad.ingestion.models import FramePacket, IngestionError, VideoMetadata
from aegis_ad.ingestion.video_source import VideoSource

__all__ = [
    "FramePacket",
    "IngestionError",
    "VideoMetadata",
    "VideoSource",
]
