# Face Recognition System

A production-grade face recognition system with real-time video processing, anti-spoofing, attendance tracking, web dashboard, and full CLI control.

## Features

**Core Recognition**
- Real-time face recognition from webcam
- Multi-image per person support (flat files or per-person folders)
- Persistent encoding cache (no re-encoding on restart)
- Optional FAISS indexing for large face databases
- Quality checks (blur, brightness, size) and face alignment
- Configurable recognition threshold and detection model (HOG/CNN)

**Anti-Spoofing**
- Liveness detection via eye blink tracking (EAR)
- Texture analysis to detect photos/screens
- Color space analysis for presentation attack detection

**Tracking & Analytics**
- Centroid-based face tracking across frames
- SQLite database for structured detection history
- Per-person detection statistics

**Attendance System**
- Automatic check-in/check-out with configurable cooldown
- CSV and Excel export
- Date range queries

**Notifications**
- Email alerts for unknown face detection
- Configurable cooldown between alerts
- Daily summary emails

**Web Dashboard**
- Live video feed in browser
- Face management (add/remove people via upload)
- Detection history and statistics
- Attendance viewer with export

**Developer Experience**
- Full CLI with argparse (threshold, camera, model, feature toggles)
- YAML configuration file
- Modular architecture (recognition/, utils/, api/)
- 40 pytest tests

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If `dlib` fails to install, install `cmake` first:
```bash
pip install cmake
```

### 2. Add Known Faces

Place images in `known_faces/`. Two layouts are supported:

**Flat (one image per person):**
```
known_faces/
  bhuvan.jpeg
  virat.jpeg
```

**Folder (multiple images per person):**
```
known_faces/
  bhuvan/
    bhuvan_1.jpg
    bhuvan_2.jpg
  virat/
    virat_1.jpg
```

### 3. Run

```bash
# Default mode (OpenCV window)
python main.py

# With custom settings
python main.py --camera 1 --threshold 0.5 --model cnn

# Web dashboard
python main.py --web --port 8080

# Disable specific features
python main.py --no-liveness --no-tracking

# Export attendance
python main.py --export-attendance 2026-02-08
```

## Configuration

All settings are in `config.yaml`. CLI arguments override config values.

```yaml
recognition:
  threshold: 0.6        # 0.4 (strict) to 0.7 (lenient)
  model: "hog"          # "hog" (CPU) or "cnn" (GPU)
  skip_frames: 2        # Process every Nth frame

camera:
  index: 0

liveness:
  enabled: true

attendance:
  enabled: true
  cooldown_minutes: 30
  export_format: "csv"

web:
  enabled: false
  port: 5000

notifications:
  enabled: false
  email:
    smtp_server: "smtp.gmail.com"
    sender: ""
    password: ""
    recipients: []
```

## Controls (OpenCV mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save session |
| `r` | Register new face from webcam |
| `a` | Print today's attendance |

## Visual Indicators

| Box Color | Meaning |
|-----------|---------|
| Green | Recognized person (name + confidence) |
| Red | Unknown person |
| Orange | Possible spoof detected |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
face-recognition_project/
  main.py                  # Main application + orchestrator
  cli.py                   # CLI argument parser
  config.yaml              # Configuration file
  requirements.txt         # Dependencies
  webcam_test.py           # Camera diagnostic tool
  recognition/
    engine.py              # Core recognition engine
    liveness.py            # Anti-spoofing detection
    tracker.py             # Face tracking across frames
  utils/
    config.py              # YAML config loader
    database.py            # SQLite backend
    encoding_cache.py      # Persistent encoding cache
    attendance.py          # Attendance system + export
    notifications.py       # Email alert system
  api/
    web_app.py             # Flask web dashboard
  templates/               # HTML templates for web UI
  static/                  # Static assets
  tests/                   # pytest test suite (40 tests)
  known_faces/             # Reference images
  data/                    # Runtime data (DB, cache, exports)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Video capture and image processing |
| face-recognition | Face detection and encoding (dlib backend) |
| numpy | Numerical operations |
| PyYAML | Configuration file parsing |
| Flask | Web dashboard |
| pytest | Testing |
| faiss-cpu (optional) | Fast nearest-neighbor for large databases |
| openpyxl (optional) | Excel attendance export |

## License

This project is for educational purposes.
