# AEGIS-AD — Preventive AI Safety System

AEGIS-AD is an AI-assisted safety and investigation platform designed to detect safety events in CCTV video feeds, track persons and vehicles, raise alerts, and retrieve supporting video evidence.

Development is organized into strictly controlled stages, starting with Stage 0 (Project Foundation).

---

## Stage 0: Project Foundation

Stage 0 sets up the core Python 3.11 environment, configuration management, logging, health check, and test suite. Ingestion, detection, tracking, and vision pipelines are intentionally excluded at this stage.

### Project Layout

```text
aegis-ad/
├── AGENTS.md                  # Project rules and engineering guidelines
├── README.md                  # Project documentation
├── pyproject.toml             # Project metadata, dependencies, and pytest configuration
├── .env.example               # Environment variable override templates
├── config/
│   └── default.yaml          # Default configuration parameters
├── src/
│   └── aegis_ad/
│       ├── __init__.py        # Package version and exports
│       ├── __main__.py        # CLI entrypoint and health check
│       ├── settings.py        # Configuration loading and validation
│       ├── logging_setup.py   # Logging initialization
│       ├── ingestion/         # Stage 1: Video ingestion
│       │   ├── __init__.py    # Ingestion exports
│       │   ├── __main__.py    # Ingestion CLI smoke test
│       │   ├── models.py      # VideoMetadata and FramePacket models
│       │   └── video_source.py# VideoSource reader
│       ├── detection/         # Stage 2: Object detection
│       │   ├── __init__.py    # Detection exports
│       │   ├── __main__.py    # Detection CLI smoke test
│       │   ├── models.py      # BoundingBox, Detection, DetectionResult models
│       │   └── detector.py    # YOLO detector wrapper (YOLO26n)
│       └── tracking/          # Stage 3: Per-camera object tracking
│           ├── __init__.py    # Tracking exports
│           ├── __main__.py    # Tracking pipeline CLI smoke test
│           ├── models.py      # TrackedObject, TrackedFrame, TrackingError models
│           └── tracker.py     # ObjectTracker (ByteTrack adapter)
├── tests/
│   ├── test_settings.py       # Configuration and validation tests
│   ├── test_health.py         # Health check and logging tests
│   ├── test_ingestion.py      # Video source and packet tests
│   ├── test_detection.py      # Detection data model & mock inference tests
│   └── test_tracking.py       # Tracking models, adapter, and pipeline tests
├── docs/
│   └── DEVELOPMENT_STAGES.md  # Roadmap of project stages
├── data/                      # Local video and snapshot storage
└── logs/                      # Application runtime logs
```

---

## Setup & Verification

### 1. Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Editable Package with Dev Dependencies
```powershell
python -m pip install -e ".[dev]"
```

### 3. Run Tests
```powershell
python -m pytest -q
```

### 4. Run Foundation Health Check
```powershell
python -m aegis_ad
```

### 5. Run Video Ingestion Smoke Test (Stage 1)
```powershell
python -m aegis_ad.ingestion --source data/videos/sample.mp4 --camera-id CAM_01 --max-frames 10
```

### 6. Run Object Detection Smoke Test (Stage 2)
```powershell
python -m aegis_ad.detection --source data/videos/sample.mp4 --camera-id CAM_01 --max-frames 10
```

### 7. Run Tracking Pipeline Smoke Test (Stage 3)
```powershell
python -m aegis_ad.tracking --source data/videos/sample.mp4 --camera-id CAM_01 --max-frames 10
```

---

## Licensing & Third-Party Dependencies

- **Project Licence:** AEGIS-AD is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
- **Ultralytics YOLO:** Object detection uses the [Ultralytics](https://github.com/ultralytics/ultralytics) Python package and pretrained YOLO models, licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Model weights are downloaded dynamically on demand and are not stored in source control.
