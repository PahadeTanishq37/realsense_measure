# RealSense Measure

3D reconstruction and non-contact anthropometric measurement pipeline using a single Intel RealSense D455 depth camera.

The system performs a handheld camera sweep around a still human subject (or target object) to capture multi-view RGB-D frames, align them into a coherent 3D point cloud, and compute anatomically accurate head and body measurements (circumferences, cranial breadths, lengths, and facial features).

---

## How It Works

The pipeline executes five sequential stages:

| Stage | Name | Description |
|---|---|---|
| **1** | **Capture** | Live camera streaming with motion-based auto-capture or manual trigger. Can also record raw uncompressed frames or replay existing captures without camera hardware. |
| **2** | **Isolate** | Per-frame depth-range cropping, background plane segmentation, statistical outlier removal, and target region clustering (e.g. head/torso separation). |
| **3** | **Register** | Pairwise and multiway alignment using facial landmarks (MediaPipe), Colored ICP, FPFH feature matching + RANSAC, and pose graph optimization. |
| **4** | **Fuse** | Integrates all aligned frames into a single unified point cloud via TSDF voxel integration or voxel-grid ICP fusion with streak removal and strict face-region gating. |
| **5** | **Measure** | Target-specific anatomical geometry extraction: cranial axis alignment, convex/concave slice circumference estimation, distance validation, and JSON report export. |

---

## Setup

### Requirements

- Python 3.10+
- Intel RealSense D455 (only needed for live capture; replay works entirely without camera hardware or pyrealsense2)

Install runtime dependencies:
`ash
pip install -r requirements.txt
`

For development and running tests:
`ash
pip install -r requirements-dev.txt
`

---

## Usage

### 1. Live Capture & Scanning

Scan a human head target with live visualizer feedback:
`ash
python -m realsense_measure.main --target head
`

Common capture options:
- --interval 1.5: Capture interval in seconds (default: 2.0).
- --manual-capture: Trigger frame capture manually with SPACE.
- --no-viz: Headless execution (disable Open3D popup display windows).
- --out <folder>: Destination directory for scan outputs.

### 2. Recording Raw Captures

Record lossless raw frames (color PNGs, float32 depth .npy in metres, and camera intrinsics meta.json) before Stage 2 isolation:
`ash
python -m realsense_measure.main --target head --save-raw raw_capture_01
`

### 3. Offline Replay (No Camera Hardware Required)

Replay a recorded capture without requiring pyrealsense2 or a physical camera:
`ash
python -m realsense_measure.main --target head --replay raw_capture_01
`

Replay with Head-Only Registration:
`ash
python -m realsense_measure.main --target head --replay raw_capture_01 --head-only-registration
`

### 4. Replay Batch Comparison

Benchmark multiple registration and fusion strategies against the same recording:
`ash
python -m realsense_measure.main --target head --replay-batch raw_capture_01 --configs baseline,multiway+tsdf,head_only+multiway+tsdf
`
This runs each configuration in an isolated subdirectory and exports an aggregated summary to comparison_summary.csv (config_name, 
egistration_tier_counts, mean_pair_rmse_mm, median_pair_rmse_mm, max_pair_rmse_mm, edges, points_in_stage4, usion_method, stage5_measurement).

---

## Advanced Flags

- --head-only-registration: Focuses pairwise registration on the head region (spherical ROI around facial landmarks clipped above chin/neck or propagated across poses) while applying final transforms to full clouds.
- --face-quality: Enables strict face-region coverage and quality metrics gating.
- --multiway: Builds a multiway pose graph and executes global loop optimization.
- --loop-closure: Evaluates non-consecutive candidate pairs for loop closure.
- --tsdf: Reconstructs using TSDF volumetric integration instead of point cloud fusion.

---

## Running Tests

Run the full automated test suite from the repository root:
`ash
python -m pytest
`

To run a specific test file:
`ash
python -m pytest tests/test_record_replay_and_head_roi.py
`

---

## Project Layout

`
.
├── README.md               # Main project documentation
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development & test dependencies
├── .gitignore              # Git ignore rules for captures, outputs, and caches
│
├── realsense_measure/      # Core pipeline package
│   ├── main.py             # CLI entry point
│   ├── pipeline.py         # 5-stage scan orchestration
│   ├── config.py           # Dataclass configurations per stage
│   ├── preprocessing.py    # Point cloud filtering and isolation
│   ├── registration.py     # Pairwise, landmark, and pose graph alignment
│   ├── reconstruction.py   # Cloud fusion and TSDF integration
│   ├── landmarks.py        # MediaPipe landmark extraction & 3D projection
│   ├── visualizer.py       # Open3D visualization helpers
│   ├── camera/             # Camera drivers and raw capture/replay I/O
│   │   ├── frame.py        # RGBDFrame abstraction and point cloud conversion
│   │   ├── recording.py    # Raw capture serialization & batch replay
│   │   └── realsense_capture.py # pyrealsense2 camera interface
│   └── targets/            # Target-specific segmentation & measurement
│       ├── base.py         # Target base class and registry
│       ├── head.py         # Human head segmentation & anthropometrics
│       └── box.py          # Box plane segmentation & OBB measurement
│
├── tools/                  # Offline analysis, viewer, and camera diagnostics
│   ├── review_scan.py      # Interactive Open3D visualizer for completed scans
│   ├── evaluate_head_scan_offline.py # Metrics evaluation on saved scans
│   └── camera_check.py     # RealSense live hardware diagnostic check
│
├── tests/                  # Automated test suite
│   ├── conftest.py         # Pytest sys.path setup
│   └── known_issues/       # Documented bug repro test cases
│
└── docs/                   # Documentation and project assets
    └── presentation.html   # Interactive presentation slides
`
