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

## Stage 3: Per-Camera Tracking (Current)
- **Goal:** Assign stable track IDs to objects within a single camera stream using ByteTrack.
- **Scope:**
  - Tracking data models (`TrackedObject`, `TrackedFrame`, `TrackingError`).
  - `ObjectTracker` adapter wrapping `ultralytics.trackers.byte_tracker.BYTETracker`.
  - Per-camera tracker lifecycle with explicit `reset()` for new video or stream sessions.
  - Camera-ID binding to prevent silent cross-camera state corruption.
  - CLI pipeline smoke test: `VideoSource → YOLODetector → ObjectTracker` (`python -m aegis_ad.tracking`).
  - Configurable tracker thresholds in `config/default.yaml` under `tracking:`.
- **Rules:** No cross-camera Re-ID, event rules, alerts, risk scoring, or evidence storage.
  track_id values are per-camera and per-lifecycle only; they do not represent
  cross-camera or globally unique identities.

---

## Stage 3: Per-Camera Tracking (Planned)
- **Goal:** Maintain stable track IDs within individual camera streams.
- **Scope:**
  - Per-camera tracking integration.
  - Track trajectory and state maintenance.

---

## Stage 4: Safety Event Engine & Rules (Planned)
- **Goal:** Configurable rule evaluation against tracks and detections.
- **Scope:**
  - Zone boundary definitions (restricted areas).
  - Dwell time / loitering rules.
  - Structured event creation with timestamps, camera ID, track ID, and confidence.

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
