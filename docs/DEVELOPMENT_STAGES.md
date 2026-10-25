# AEGIS-AD Development Stages

This document outlines the strictly controlled development stages for AEGIS-AD.
Each stage has explicit goals, deliverables, and acceptance criteria. A stage must be completed, verified, and approved before the next begins.

---

## Stage 0: Project Foundation (Current)
- **Goal:** Establish a clean, testable repository foundation.
- **Deliverables:**
  - Standard `src/aegis_ad` package structure.
  - Configuration loader (`config/default.yaml`) with validation and environment overrides.
  - Standard logging setup (`src/aegis_ad/logging_setup.py`).
  - Simple health-check entrypoint: `python -m aegis_ad`.
  - Automated test suite covering settings, logging, and health check.
- **Rules:** No CCTV video ingestion, object detection, tracking, or advanced AI features.

---

## Stage 1: Video Ingestion (Planned)
- **Goal:** Robust frame capture and streaming abstraction.
- **Scope:**
  - Local MP4 and video file source abstraction.
  - Basic RTSP stream connector with error handling and reconnection logic.
  - Frame generator with proper timestamping and resource management.

---

## Stage 2: Object Detection (Planned)
- **Goal:** Inference pipeline for safety-relevant objects.
- **Scope:**
  - Standard detection interface (persons, vehicles).
  - Model loader and batch/frame processor.
  - Structured detection outputs (bounding boxes, class labels, confidence scores).

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
