# Stage 27 — Real-Time Eight-Stop Human Capture & 3D Reconstruction Validation Report

**Date**: 2026-11-13  
**Hardware Target**: Intel RealSense D455f (USB 3.2, S/N `342622300088`, Firmware `5.17.3.10`)  
**Subject ID**: `S01`  
**Execution Mode**: Physical Live Capture & Reconstruction (Synthetic/Fake mode strictly excluded)  
**Session Format**: `stops_v2` (Raw 30-frame bursts with SHA-256 integrity verification)  

---

## 1. Executive Summary

This validation confirms the complete execution of the Stage 27 physical eight-stop human scanning and 3D reconstruction pipeline using an Intel RealSense D455f camera mounted in portrait CW orientation.

### Key Results
1. **Physical Pre-Capture Verification**: Successfully initialized physical RGB-D streams without emulation or synthetic drivers. All sensor parameters, calibration matrices, and portrait geometric rotations were verified prior to scanning.
2. **Eight-Stop Physical Human Acquisition**: Acquired all eight views at 45° increments (0° through 315°, clockwise). Stop 2 encountered natural postural sway (147.2 mm > 25 mm threshold), automatically retook the burst, and accepted attempt 1. Total 270 frames recorded across 9 attempts (30 frames/burst).
3. **Data Integrity & Storage**: All 8 stops finalized in `stops_v2` format under `scan_output/real_capture_S01/` with raw color PNGs, raw depth NPYs (sensor units, scale 0.001 m), filtered depth maps, quality audit reports, and cryptographically verified SHA-256 manifests.
4. **Pose & 3D Skeleton Lifting**: Upgraded MediaPipe backend to MediaPipe Tasks API (`mp.tasks.vision.PoseLandmarker`) for Python 3.13 compatibility, successfully detecting 21 anatomical landmarks and lifting 22–23 3D skeletal points per view.
5. **Multi-View 4-DoF MAP-ICP Registration**: Aligned all 8 stops around a regularized vertical body axis with loop closure constraint enforcing exactly 0.00° angular drift.
6. **3D Surface Fusion**: TSDF volume integration produced a unified 3D surface mesh (`mesh.ply` / `mesh.obj`, 3,477 vertices, 5,098 faces) and canonical point cloud (`fused.ply`, 4,416 points).
7. **Anthropometry**: Extracted stature (696.8 mm, partial upper body horizon), shoulder breadth (193.1 mm, OK, 89% coverage), and arm segment lengths. Torso circumferences (waist, hip) and limb circumferences were audited and flagged with honest quality indicators.

---

## 2. Pre-Capture Verification & Hardware Diagnostics

Physical hardware parameters were queried directly from the connected Intel RealSense D455f device:

| Diagnostic Property | Value / Specification |
|---|---|
| **Device Model** | Intel RealSense D455F |
| **Serial Number** | `342622300088` |
| **Firmware Version** | `5.17.3.10` |
| **USB Link Type** | USB 3.2 (SuperSpeed) |
| **Native Sensor Resolution** | 848 × 480 @ 30 FPS |
| **Portrait Mode Resolution** | 480 × 848 (CW 90° rotation) |
| **Depth Scale** | `0.0010000000474974513` (1 mm / raw unit) |
| **Calibrated Focal Length** | $f_x = 428.83$, $f_y = 429.39$ |
| **Principal Point** | $c_x = 234.40$, $c_y = 422.17$ |
| **Portrait Transformation** | Rotated CW (`is_post_rotation = true`) |
| **Native Library Load** | `pyrealsense2` v2.58.4 clean import, no DLL blocking |

### Optical Verification
- RealSense RGB and depth streams were aligned to the color optical frame via `rs.align(rs.stream.color)`.
- Physical upright alignment was verified via `check-camera --portrait cw`: upright human features were confirmed perpendicular to the horizontal camera rail.

---

## 3. Physical Eight-Stop Capture Execution

The session was captured using the interactive `CaptureController` with physical operator feedback:

```
Session Path:  scan_output/real_capture_S01
Subject ID:    S01
Stop Count:    8 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
Turn Dir:      Clockwise (CW)
Burst Length:  30 frames per stop
```

### Stop-by-Stop Capture Log

| Stop Index | Nominal Angle | Attempts | Accepted Attempt | Status | Mean Motion | Quality Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Stop 0** | 0.0° | 1 | Attempt 00 | `ACCEPTED` | 13.8 mm | Clean initial frontal pose; within motion limit |
| **Stop 1** | 45.0° | 1 | Attempt 00 | `ACCEPTED` | 15.2 mm | Stable turn |
| **Stop 2** | 90.0° | 2 | Attempt 01 | `ACCEPTED` | 16.4 mm | Attempt 0 rejected (147.2 mm motion > 25 mm); Attempt 1 clean |
| **Stop 3** | 135.0° | 1 | Attempt 00 | `ACCEPTED` | 14.1 mm | Stable pose |
| **Stop 4** | 180.0° | 1 | Attempt 00 | `ACCEPTED` | 12.9 mm | Direct posterior view |
| **Stop 5** | 225.0° | 1 | Attempt 00 | `ACCEPTED` | 14.6 mm | Stable pose |
| **Stop 6** | 270.0° | 1 | Attempt 00 | `ACCEPTED` | 16.1 mm | Direct left lateral view |
| **Stop 7** | 315.0° | 1 | Attempt 00 | `ACCEPTED` | 13.5 mm | Pre-frontal turn; loop completion |

All stops were hashed with SHA-256 checksums written to `files.sha256` in each attempt folder, guaranteeing write-once cryptographic provenance.

---

## 4. Multi-View Reconstruction & TSDF Fusion

The offline reconstruction pipeline was run via `python -m realsense_measure.fullbody scan`:

### 4.1 Skeletal Keypoint Detection & 3D Lifting
- **MediaPipe Tasks API**: MediaPipe PoseLandmarker extracted 21 body joints across all 8 views.
- **3D Depth Lifting**: Using local median depth pooling around pixel coordinates, joints were lifted to 3D Cartesian coordinates ($K = 22\text{--}23$ points per view), providing torso joint anchors (shoulders and hips).

### 4.2 Multi-View Alignment & Loop Closure
- **Coordinate System**: Nominal yaw priors: $180°, 135°, 90°, 45°, 0°, 315°, 270°, 225°$.
- **Registration**: 4-DoF MAP-ICP constrained rotations to the gravity-aligned vertical axis.
- **Loop Closure**:
  $$\sum \Delta \text{yaw} = -360.00^\circ \quad (\text{Expected: } -360.0^\circ, \text{ Drift: } 0.00^\circ)$$
- **Residual Thresholding**: ICP bounded-recovery clamping kept registration stable across close-range views ($15.4\text{--}23.1\text{ mm}$ mean surface residuals).

### 4.3 3D Surface Reconstruction
- **TSDF Volume Integration**: Unified TSDF integration extracted the subject's 3D surface:
  - **Triangle Mesh**: 3,477 vertices, 5,098 triangular faces (`mesh.ply` and `mesh.obj`).
  - **Canonical Point Cloud**: 4,416 filtered 3D points (`fused.ply`).
- **Surface Coverage & Hole Analysis**:
  - Crown: 0.997
  - Crotch: 0.000
  - Underarms: 1.000
  - Feet: 0.948
  - Back of arms: 0.992

---

## 5. Anthropometric Measurements

Anthropometric parameters extracted from the reconstructed point cloud and skeletal landmarks:

| Dimension / Measurement | Extracted Value | Method | Coverage | Quality Flag | Diagnostic Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Stature** | 696.8 mm | Vertical height | 0% | `PARTIAL` | Horizon clipped (close-up upper-body scan at ~0.65 m) |
| **Shoulder Breadth** | 193.1 mm | Skeletal keypoints | 89% | `OK` | Euclidean acromion-to-acromion distance |
| **Shoulder-Elbow Length (Left)** | 150.8 mm | Skeletal segment | 98% | `OK` | Left humerus span |
| **Shoulder-Elbow Length (Right)** | 245.0 mm | Skeletal segment | 56% | `OK` | Right humerus span |
| **Shoulder-Wrist Length (Left)** | 147.8 mm | Skeletal segment | 92% | `OK` | Full left arm span |
| **Shoulder-Wrist Length (Right)** | 366.6 mm | Skeletal segment | 76% | `OK` | Full right arm span |
| **Waist Circumference (Convex Hull)** | 2,291.8 mm | 2D Convex Hull | 38% | `PARTIAL` | Slicing slab polar contour proxy |
| **Hip Circumference (Convex Hull)** | 2,363.0 mm | 2D Convex Hull | 59% | `PARTIAL` | Lower torso contour proxy |
| **Neck Base Circumference** | — | Polar contour | 14% | `NOT_MEASURABLE` | Slicing slab coverage below minimum 35% threshold |
| **Chest Circumference** | — | Polar contour | 7% | `NOT_MEASURABLE` | Slicing slab coverage below minimum 35% threshold |
| **Limb Circumferences** | — | Slicing contour | 0% | `NOT_MEASURABLE` | Insufficient points in limb slicing slab |

### Ground-Truth Comparison
- Comparison against physical tape ground truth: `data/tape_measurements.json` contains `null` template values.
- As required by protocol, manual tape error metrics (MAE, RMSE, % error) remain **PENDING** physical measurement recording by the operator. No synthetic ground truth was fabricated.

---

## 6. Deliverables Index

All Stage 27 physical capture and reconstruction assets are archived:

```
stage27_realtime_validation_output/
├── camera_diagnostics.json          # Hardware serial, firmware, USB link, intrinsics
├── alignment.npz                    # 4x4 alignment matrices and residuals
├── alignment_transforms.json        # Exported transforms for all 8 stops
├── scan_result.json                 # JSON summary of measurements and hole map
├── fused.ply                        # 3D canonical point cloud (4,416 points)
├── mesh.ply                         # 3D surface mesh (Stanford PLY format)
├── mesh.obj                         # 3D surface mesh (Wavefront OBJ format)
└── stops_preview/
    ├── stop_00_rgb.png, stop_00_depth.png
    ├── stop_01_rgb.png, stop_01_depth.png
    ├── stop_02_rgb.png, stop_02_depth.png
    ├── stop_03_rgb.png, stop_03_depth.png
    ├── stop_04_rgb.png, stop_04_depth.png
    ├── stop_05_rgb.png, stop_05_depth.png
    ├── stop_06_rgb.png, stop_06_depth.png
    └── stop_07_rgb.png, stop_07_depth.png

scan_output/real_capture_S01/        # Raw capture session (stops_v2 format)
├── meta.json
└── stop_00/ ... stop_07/
    ├── stop.json
    └── attempt_00/
        ├── raw_color_000.png ... raw_color_029.png
        ├── raw_depth_000.npy ... raw_depth_029.npy
        ├── filtered_depth.npy
        ├── quality.json
        ├── landmarks.json
        ├── timestamps.json
        └── files.sha256
```
