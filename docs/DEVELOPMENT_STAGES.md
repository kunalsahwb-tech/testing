# AEGIS-AD Development Stages

This document outlines the strictly controlled development stages for AEGIS-AD.
Each stage has explicit goals, deliverables, and acceptance criteria. A stage must be completed, verified, and approved before the next begins.

---

## Stage 0: Project Foundation (Completed)
- **Goal:** Establish a clean, testable repository foundation.
- **Deliverables:**
  - Standard `src/aegis_ad` package structure.
  - Configuration loader (`config/default.yaml`) with validation and environment overrides.
  - Standard logging setup (`src/aegis_ad/logging_setup.py`).
  - Simple health-check entrypoint: `python -m aegis_ad`.
  - Automated test suite covering settings, logging, and health check.
- **Rules:** No CCTV video ingestion, object detection, tracking, or advanced AI features.

---

## Stage 1: Video Ingestion (Completed)
- **Goal:** Robust frame capture and streaming abstraction.
- **Scope:**
  - Local MP4 video file source abstraction (`VideoSource`).
  - FramePacket and VideoMetadata data models.
  - Strict FPS validation and frame-index-based timestamping.
  - Automatic resource cleanup.

---

## Stage 2: Object Detection (Completed)
- **Goal:** Inference pipeline for safety-relevant objects using pretrained YOLO26n.
- **Scope:**
  - Detection data models (`BoundingBox`, `Detection`, `DetectionResult`).
  - YOLODetector wrapping Ultralytics YOLO26n.
  - Filter for target classes (person, bicycle, car, motorcycle, bus, truck).
  - CLI detection smoke test (`python -m aegis_ad.detection`).
- **Rules:** No tracking, Re-ID, event rules, alerts, or custom model training.

---

## Stage 3: Per-Camera Tracking (Completed)
- **Goal:** Assign stable track IDs to objects within a single camera stream using ByteTrack.
- **Scope:**
  - Tracking data models (`TrackedObject`, `TrackedFrame`, `TrackingError`).
  - `ObjectTracker` adapter wrapping `ultralytics.trackers.byte_tracker.BYTETracker`.
  - Per-camera tracker lifecycle with explicit `reset()` for new video or stream sessions.
  - Camera-ID binding to prevent silent cross-camera state corruption.
  - CLI pipeline smoke test: `VideoSource → YOLODetector → ObjectTracker` (`python -m aegis_ad.tracking`).
  - Configurable tracker thresholds in `config/default.yaml` under `tracking:`.

---

## Stage 4: Safety Event Engine & Rules (Current)
- **Goal:** Detect restricted area entries and dwell-time exceedances from tracked persons.
- **Scope:**
  - Event data models (`RestrictedZone`, `SafetyEvent`, `EventConfigurationError`).
  - Ray-casting point-in-polygon geometry with boundary edge/vertex inclusion.
  - Reference point calculation using normalized bottom-center point of bounding box.
  - `EventEngine`:
    - `restricted_area_entry`: Emitted when an observed person transitions from outside to inside a zone.
    - `dwell_time_exceeded`: Emitted once per continuous episode when a person dwells >= configured threshold.
    - Tracking gap tolerance via `absence_grace_seconds`.
    - Camera-specific zone isolation and person class ID filtering.
  - Configuration in `config/default.yaml` under `events:`.
  - CLI safety event smoke test: `VideoSource → YOLODetector → ObjectTracker → EventEngine` (`python -m aegis_ad.events`).
- **Rules:** No risk scoring, alert delivery, evidence storage, Re-ID, or web interface.

---

## Stage 5: Evidence Management & Alerts (Planned)
- **Goal:** Alert generation with traceable media evidence.
- **Scope:**
  - Event snapshots and clip clipping/referencing.
  - Alert notification formatting and evidence indexing.

---

## Stage 6: Operator Interface & Future Extensions (Planned)
- **Goal:** Dashboard review and advanced investigation workflows.
- **Scope:**
  - Operator dashboard / API.
  - Future AI investigation tool integration and cross-camera Re-ID.
