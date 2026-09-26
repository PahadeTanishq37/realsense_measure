# realsense_measure

**realsense_measure** captures several frames of an object from different angles using a single, fixed Intel RealSense D455f depth camera, registers and fuses those frames into one unified 3-D point cloud, and reports the object's real-world dimensions. The pipeline is designed to start with rigid objects such as boxes and packages, then extend progressively to more complex targets — heads (for hat/helmet sizing) and eventually full bodies (for clothing and ergonomic measurement) — without touching the core capture, registration, or fusion code.

---

## How it works

The pipeline runs five sequential stages:

| Stage | What happens |
|---|---|
| **1. Capture** | Live camera feed shown on screen; automatically grabs frames at configurable intervals, ENTER to finish |
| **2. Isolate** | Each raw frame is denoised, the dominant plane (table/background) is removed, and the largest cluster (the object) is kept |
| **3. Register** | All isolated frames are aligned into one coordinate system using FPFH feature matching + RANSAC for the global pose, then refined with point-to-plane ICP |
| **4. Fuse** | All aligned frames are merged into one dense point cloud, then cleaned again |
| **5. Measure** | The target-specific logic fits an oriented bounding box (for a box), applies bias-corrected percentile trimming, and writes dimensions + a JSON result |

---

## Setup

Install dependencies on the machine where the D455f is physically plugged in:

```bash
pip install -r requirements.txt
```

> **Note:** `pyrealsense2` communicates directly with the camera over USB.
> It must run on the machine the D455f is connected to — remote/SSH-only
> environments will not work unless USB forwarding is configured.

---

## Usage

```bash
# Scan a box with default settings (Open3D stage windows enabled, 4 mm voxel)
python main.py

# Specify target type and output directory
python main.py --target box --out my_scan

# Skip the Open3D popup windows (headless / scripted use)
python main.py --no-viz

# Use a finer voxel grid for higher-detail scans (slower)
python main.py --voxel 0.003

# Change automatic frame capture interval (e.g. 0.5s or 2.0s)
python main.py --interval 0.5
```

### Capture controls

| Key | Action |
|---|---|
| *(Timer)* | Frames are captured automatically at `capture_interval_s` (default 3.0s) |
| `SPACE` | Force immediate capture of current frame |
| `ENTER` | Finish capture and start processing (requires ≥ 2 frames) |
| `ESC` | Abort the scan |

---

## Capture tips

### Show genuinely different angles — not small wobbles

Aim to capture **at least 3–4 views that expose different faces of the object**: front, side, top, and ideally a diagonal. Slight wobbles or small rotations around the same axis are not enough.

**Why this matters:** a depth camera only sees the surface currently facing it. If you never tilt the object to show its top face, the pipeline has no data for that dimension and the recovered height will come out undersized — sometimes by several centimetres. Every face you want measured must appear in at least one frame.

### Distance

Keep the object **0.3 m – 1.0 m** from the camera. The D455f's depth accuracy degrades noticeably below ~0.3 m and beyond ~1.2 m.

### Background

Use a **plain, contrasting background** — a clean table surface works well. The isolation step removes the dominant plane (assumed to be the background) and then clusters what remains. A cluttered or similarly-coloured background confuses this step and may leave background geometry in the object cloud.

---

## Validating without hardware

Run the synthetic end-to-end test to verify the full algorithm pipeline — **no camera or display window required**:

```bash
cd realsense_measure
python test_synthetic.py
```

`test_synthetic.py` builds a synthetic 300 × 200 × 120 mm box, simulates 8 camera views with realistic sensor noise (1.5 mm σ), runs the full isolate → register → fuse → measure chain, and asserts that all three recovered dimensions are within **15 mm** of ground truth. In practice, typical recovered error is **0–10 mm per dimension** at the simulated noise level.

A `RESULT: PASS` line at the end means the algorithm logic is intact on your machine even without the physical sensor.

---

## Project layout

```
realsense_measure/
├── main.py                     Command-line entry point (argparse flags, calls ScanPipeline)
├── pipeline.py                 Orchestrates all 5 stages; the only file that calls the others in sequence
├── config.py                   All tunable parameters in one place (dataclasses per stage)
├── preprocessing.py            Per-frame: downsample, denoise, remove background plane, cluster
├── registration.py             Multi-view alignment: FPFH+RANSAC global registration → ICP refinement
├── reconstruction.py           Merge all aligned frames into one fused point cloud
├── visualizer.py               Open3D helpers: colour frames, show windows, draw OBB wireframes
├── test_synthetic.py           Synthetic end-to-end test; runs with no camera or display
├── requirements.txt            Pinned-minimum Python dependencies
├── camera/
│   ├── __init__.py
│   └── realsense_capture.py   ONLY file that imports pyrealsense2; wraps D455f stream + filters
└── targets/
    ├── __init__.py
    ├── base.py                 Abstract Target class + @register_target decorator + get_target()
    └── box.py                  BoxTarget: plane-strip segmentation + trimmed-OBB measurement
```

---

## Extending to heads, then bodies

Only **`targets/box.py`** is object-specific. The rest of the pipeline — capture, preprocessing, registration, fusion — is completely target-agnostic.

### Adding a head target

1. Create `targets/head.py` and implement the `Target` interface:

   ```python
   from targets.base import Target, register_target

   @register_target
   class HeadTarget(Target):
       name = "head"

       def segment(self, pcd):
           # Different logic from box: depth-range crop to isolate the head
           # volume, then crop to the face region using a bounding-sphere
           # heuristic rather than a plane-removal pass.
           ...

       def measure(self, pcd):
           # Different logic from box: compute circumference from a horizontal
           # cross-section, or measure landmark distances (e.g. front-to-back
           # vs. ear-to-ear) from the oriented principal axes.
           # Return the same dict contract (JSON-serialisable + "geometry_for_viz").
           ...
   ```

2. Import it in `targets/__init__.py` (one line) so the `@register_target` decorator fires at startup.

3. Add `"head"` to `--target`'s `choices` list in `main.py`.

That is the entire change. `pipeline.py`, `registration.py`, `reconstruction.py`, `preprocessing.py`, and `camera/realsense_capture.py` require **zero modifications**.

The same pattern applies for `targets/body.py`.

---

## Known limitations

- **Low-detail frames can mis-align.** After Stage 3, the pipeline prints per-frame `fitness` and `rmse` values. A fitness score well below **~0.5** on a frame after frame 0 indicates a weak alignment — that angle probably overlapped poorly with the growing reference cloud. Recapture it showing a cleaner, distinct face of the object.

- **Reflective, transparent, and very dark surfaces** are problematic for any structured-light or time-of-flight depth sensor, including the D455f. Matte, diffusely-reflecting surfaces give the most reliable depth measurements. If accuracy matters, consider applying temporary matte spray to highly reflective objects.

- **Single fixed camera.** Because the camera does not move, the reconstruction relies entirely on the user rotating the object. Uneven or rushed rotation patterns lead to gaps in surface coverage and weaker registration for the frames that cover those gaps.
