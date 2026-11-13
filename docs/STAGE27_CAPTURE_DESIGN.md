# Stage 27 — Live Eight-Stop Capture Controller: Design Document

**Branch**: `fullbody-v2`  
**Date**: 2026-11-13  
**Status**: READ-ONLY design — no source files modified.  
**Hardware**: Intel RealSense D455f, fixed tripod, subject rotates 8 stops × 45°, A-pose, burst 30–45 frames/stop.

---

## Q1 — REUSE MAP

For each capture requirement: **Fully covered**, **Partly covered (gap noted)**, or **Does not exist**.

| Requirement | Coverage | Existing code | Gap / Note |
|---|---|---|---|
| **RGB validity** (non-None, non-black frame) | Partly | `read()` returns `(None, None)` on dropped frame [`realsense_capture.py:467-468`]. `evaluate_pre_save()` checks `color is None` [`capture_session.py:338-346`]. | No per-pixel saturation / all-black check. No "exposure stable" pre-burst warm-up gate. |
| **Depth validity %** (fraction of non-zero, in-range pixels) | Fully | `CaptureQualityChecker.compute_depth_stats()` [`capture_session.py:239-276`] and `evaluate_pre_save()` WARN at < 30% valid fraction [`capture_session.py:364-378`]. `FullBodyConfig.depth_min_m = 0.40`, `depth_max_m = 2.60` [`fullbody/config.py:40-41`]. | — |
| **Human coverage** (body silhouette in frame) | Partly | `evaluate_pre_save()` checks boundary clipping at 3 px margin using foreground mask [`capture_session.py:380-408`]. `extract_subject_mask()` isolates the subject silhouette [`capture_motion.py:38-66`]. | No explicit "coverage fraction" threshold (what fraction of image is occupied by the subject). No minimum-pixel-count guard. |
| **Invalid-depth %** (zero / NaN / out-of-range pixel fraction) | Fully | `compute_depth_stats()` counts zeros, NaNs, out-of-range pixels and reports all three [`capture_session.py:246-276`]. | — |
| **Temporal stability** (depth stable across burst) | Fully | `evaluate_burst_stillness()` computes peak-to-peak silhouette + 3D depth centroid motion across the burst; flags if > 10 mm [`capture_motion.py:140-201`]. Also encoded in `BurstMotionResult.is_flagged` [`capture_motion.py:25-35`]. | — |
| **MediaPipe landmark visibility** | Partly | `MediaPipePoseDetector.detect()` returns per-joint visibility and confidence [`keypoints.py:85-129`]. `lift_keypoints_to_3d()` rejects joints below `keypoint_min_confidence = 0.40` and `keypoint_min_visibility = 0.50` [`fullbody/config.py:65-66`; `keypoints.py:171`]. | No real-time gating of captures based on live landmark visibility during burst. |
| **Head + feet visibility** | Partly | `evaluate_pre_save()` flags `"top (head crown clipped)"` and `"bottom (lower body clipped)"` via boundary contact check [`capture_session.py:389-401`]. | Head-top check is a crude border-pixel heuristic, not a semantic check that the crown and ankles are both present. |
| **Subject motion** (inter-frame stillness during burst) | Fully | `evaluate_burst_stillness()` with 10 mm threshold, flagged in `BurstMotionResult.is_flagged` [`capture_motion.py:188-189`]. `HeadCaptureConfig.stillness_depth_thresh_m = 0.015 m` and `stillness_time_s = 0.7 s` for auto-hold trigger [`config.py:110-112`]. | Motion threshold (10 mm) is **untested** against real full-body data. |
| **Body clipping at image borders** | Partly | `evaluate_pre_save()` checks 3 px border contact per frame [`capture_session.py:380-408`]. | Only checks 3 px inward; does not distinguish intentional background vs. body clipping. |
| **Burst capture** (N consecutive frames per stop) | Partly | `HeadCaptureConfig.burst_count = 10` for head mode [`config.py:104`]. `FullBodyConfig.burst_frame_count = 30` for full-body [`fullbody/config.py:29`]. `process_stop_burst()` accepts a list of burst frames [`burst.py:24-99`]. | The controller loop that actually *collects* 30–45 frames at each stop and assembles the list **does not exist** — it must be written for Stage 27. |
| **Stillness wait** (pause until subject stops swaying before burst) | Partly | `auto_hold` / `stillness_depth_thresh_m` logic exists in `HeadCaptureConfig` [`config.py:108-112`]. `evaluate_burst_stillness()` can retroactively flag motion [`capture_motion.py:140-201`]. | No pre-burst gate that blocks the burst start until N consecutive frames satisfy the stillness threshold. |
| **Recapture** (re-do a stop if quality fails) | Does not exist | `BurstMotionResult.is_flagged` and `CaptureQualityChecker.evaluate_pre_save()` produce boolean/list flags. | No retry state machine, no per-stop recapture counter, no operator prompt to re-do a stop. |

---

## Q2 — RAW vs FILTERED

### Filter chain in `read()`

`read()` calls `_filter_depth()` on the aligned depth frame **before** conversion to numpy.  
Exact filter order ([`realsense_capture.py:326-371`]):

```
(1) DecimationFilter      — if cfg.use_decimation == True  (default: True, magnitude 2)
(2) DisparsityTransform   — depth → disparity (always applied around spatial/temporal)
(3) SpatialFilter         — if cfg.use_spatial_filter == True  (default: True)
(4) TemporalFilter        — if cfg.use_temporal_filter == True  (default: True)
(5) DisparsityTransform   — disparity → depth (always)
(6) HoleFillingFilter     — if cfg.use_hole_filling == True   (default: True)
```

Filter parameters:  
- **Spatial**: magnitude 1 (face quality) or 2 (default), alpha 0.5, delta 20.0, holes_fill 0.0 [`realsense_capture.py:113-117`]  
- **Temporal**: alpha 0.4, delta 20.0, holes_fill 3.0 (2/4-frame persistence mode) [`realsense_capture.py:119-122`]  
- **Hole-filling**: mode 1 = "farthest from around" [`realsense_capture.py:126`]

### Which method returns truly raw (unfiltered) depth

`read_with_raw_and_colorized()` returns **both** filtered and raw arrays [`realsense_capture.py:674-761`]:
- `depth_filtered_m` — post-processed (all 6 filter stages above)
- `depth_raw_m` — straight from `np.asanyarray(raw_depth_frame.get_data()) * depth_scale`, before any filter [`realsense_capture.py:727-728`]
- `depth_raw_z16` — native uint16 sensor units [`realsense_capture.py:727`]

### Effect of `depth_max_m = 1.20` on a subject at 1.3 m

`CameraConfig.depth_max_m = 1.20` m (default) **clips** a subject standing at 1.3 m — the subject's depth exceeds the range gate and is zeroed out.  
`config.py:177-180` corrects this for `target_name == "body"` by raising `depth_max_m` to 3.00 m.  
`FullBodyConfig.depth_max_m = 2.60 m` [`fullbody/config.py:41`] is applied in burst processing.  
**For Stage 27**, the controller must ensure `CameraConfig.depth_max_m >= 2.60` or `3.00` before starting the stream.

### Effect of decimation magnitude 2

Decimation halves linear resolution: 848×480 → 424×240.  
`_filter_depth()` applies decimation first; after decimation the depth frame is at half resolution.  
`read()` resizes the smaller depth array back to match color resolution using `cv2.INTER_NEAREST` [`realsense_capture.py:488-492`].  
**Intrinsics impact**: After `cv2.resize` back up, the `intrinsics_o3d` object was built from the **original** color-stream intrinsics (not the post-decimation depth intrinsics) [`realsense_capture.py:256-267`]. This means the depth values are correct (re-upsampled) but the correspondence at each pixel is block-replicated rather than sub-pixel accurate. At the full-body voxel scale (5 mm), this is acceptable.

### Proposed `CameraConfig` for full-body capture (Stage 27)

| Parameter | Proposed value | Justification |
|---|---|---|
| `width`, `height` | 848, 480 | Native landscape resolution; portrait is handled by `rotate_90`. D455 supports 848×480 natively at 30 fps. |
| `rotate_90` | `"cw"` | Portrait mount, 90° clockwise rotation (see Q3). |
| `fps` | 30 | Provides ~30 frames per 1-second burst. |
| `depth_min_m` | 0.40 | Matches D455 near-distance floor; subjects at 1.3 m are well above. |
| `depth_max_m` | 3.00 | Covers 1.3 m subject depth + background room. |
| `use_decimation` | **False** | Full-body at 5 mm voxels needs every pixel; decimation costs resolution without measurable gain. If decimation must stay on, the intrinsic swap in `start()` must be verified (see RISKS). |
| `use_spatial_filter` | True | Fills sensor dropouts on clothing/hair without destroying edges at this scale. |
| `use_temporal_filter` | True | Burst-to-burst consistency; reduces random noise. |
| `use_hole_filling` | True | Small dropouts at dark clothing seams. |

**However**, for the Stage 27 burst controller, the **raw** depth per frame should be recorded (via `read_with_raw_and_colorized()`) alongside the filtered depth. The burst stack assembled by `process_stop_burst()` should use the **filtered per-frame** depth (SDK filtering already applied) rather than the raw, because the per-frame SDK filters (spatial, temporal) provide better signal for the temporal median.

---

## Q3 — ORIENTATION

### Where 90° rotation is applied

Portrait orientation is handled by `cfg.rotate_90 ∈ {"cw", "ccw"}`.

1. **Intrinsics** are adjusted immediately after `start()` by calling `rotate_intrinsics()` from `realsense_measure.camera.rotation` [`realsense_capture.py:265-267`]:
   ```python
   if self.cfg.rotate_90 in ("cw", "ccw"):
       from realsense_measure.camera.rotation import rotate_intrinsics
       self.intrinsics_o3d = rotate_intrinsics(self.intrinsics_o3d, self.cfg.rotate_90)
   ```
2. **Image arrays** are rotated inside `read()` after conversion to numpy, via `rotate_rgbd_arrays()` [`realsense_capture.py:494-496`]:
   ```python
   if self.cfg.rotate_90 in ("cw", "ccw"):
       color_bgr, depth_m = rotate_rgbd_arrays(color_bgr, depth_m, self.cfg.rotate_90)
   ```
3. Same rotation is applied in `read_with_colorized()` [`realsense_capture.py:611-616`] and `read_with_raw_and_colorized()` [`realsense_capture.py:752-759`].

**Result**: After `read()`, both `color_bgr` and `depth_m` are in portrait orientation, and `intrinsics_o3d` is in the portrait coordinate frame. All downstream code (burst.py, keypoints.py, cli.py, io.py) therefore always operates on portrait arrays.

### How `turn_direction` and `camera_portrait_rotation` reach `meta.json`

**`save_raw_capture()`** writes `camera_config` to `meta.json` [`recording.py:83-96`]:
```json
"camera_config": {
  "rotate_90": "cw",   // getattr(cfg.camera, "rotate_90", None)
  ...
}
```
`turn_direction` is **NOT** persisted by `save_raw_capture()` — it is a `FullBodyConfig` field, not a `CameraConfig` field. `camera_portrait_rotation` is likewise a `FullBodyConfig` field, not persisted in the standard `meta.json` schema.

**`io.py` reads them back** at load time from the `meta.json` root (not `camera_config`):
```python
turn_dir = meta.get("turn_direction", cfg.turn_direction)        # io.py:107
cam_rot  = meta.get("camera_portrait_rotation", cfg.camera_portrait_rotation)  # io.py:108
```
So `io.py` expects `turn_direction` and `camera_portrait_rotation` at the **root level** of `meta.json`, but `save_raw_capture()` does not write them there. **This is a gap**: a Stage 27 controller must add both fields to the top-level `meta.json` document when saving.

### Intrinsics swap risks after rotation

| Risk | Location | Detail |
|---|---|---|
| `cx`/`cy` swap | `rotate_intrinsics()` (not read; assumed correct) | For 90° CW: new `cx = old_cy`, new `cy = W - 1 - old_cx`. Must verify `rotate_intrinsics` does this correctly. **Unverified** — `realsense_measure/camera/rotation.py` was not in the required reading list and was not read. |
| `fx`/`fy` swap | Same | For 90° CW, width and height are exchanged; fx and fy are swapped accordingly. Same unverified status. |
| `camera_config.width`/`height` in `meta.json` | `recording.py:84-85` | Writes `cfg.camera.width, cfg.camera.height` which are pre-rotation (848×480). After rotation the portrait image is 480×848. `io.py` reads `intr.width, intr.height` from `meta.json` intrinsics dict [`io.py:91-92`], not from `camera_config`. Intrinsics dict is taken from `first_frame.intrinsics` which is the already-rotated intrinsic matrix. This should be consistent, but needs verification with real frames. |
| `_load_subdirs_session()` guesses intrinsics | `io.py:180-183` | Uses `fx = fy = max(h, w)` with no reference to recorded `rotate_90`. If a portrait image (480×848) is loaded from a subdir, `max(h, w) = 848` which gives an approximately correct focal length, but cx/cy and fx/fy will not be camera-calibrated. **Unverified correctness**. |

---

## Q4 — LOADER GAPS

### Existing supported layouts

**Layout 1** (meta.json + frame files, `_load_meta_json_session()`):
```
capture_dir/
  meta.json                   # intrinsics, camera_config, turn_direction*, frames[]
  frame_00_color.png
  frame_00_depth.npy
  ...
```
`*` `turn_direction` and `camera_portrait_rotation` read from root but not written by `save_raw_capture()`.

**Layout 2** (per-stop subdirs, `_load_subdirs_session()`):
```
capture_dir/
  stop_00/
    *color*.png, *.jpg
    *depth*.npy, *depth*.png
  stop_01/
  ...
```
Only the **first** color and depth file per stop subdir is loaded [`io.py:172-176`]. Burst frames are discarded.

### Proposed Stage 27 extended layout

```
capture_dir/
  meta.json                      # root-level: turn_direction, camera_portrait_rotation,
                                 #  num_stops, nominal_angles_deg[], depth_scale,
                                 #  intrinsics{}, camera_config{}
  trial_meta.json                # trial_status: "completed"/"aborted" (existing pattern)
  stop_00/
    stop_meta.json               # stop_id, nominal_angle_deg, quality_report{},
                                 #  burst_frames_used, burst_motion_mm, accept_reject_history[]
    burst_color_000.png          # raw per-frame color (never overwritten)
    burst_color_001.png
    ...
    burst_depth_raw_000.npy      # raw (unfiltered) per-frame depth float32 m
    burst_depth_000.npy          # filtered per-frame depth (SDK-filtered) float32 m
    color.png                    # representative color (middle burst frame)
    depth.npy                    # burst-processed depth (output of process_stop_burst)
    depth_std.npy                # per-pixel temporal std from burst processing
    valid_counts.npy             # per-pixel valid-frame count
  stop_01/
  ...
```

### Schema additions to `meta.json` (root level)

```json
{
  "intrinsics": { "width": 480, "height": 848, "fx": ..., "fy": ..., "cx": ..., "cy": ... },
  "camera_config": { ... },
  "turn_direction": "cw",
  "camera_portrait_rotation": "cw",
  "depth_scale": 0.0001,
  "num_stops": 8,
  "nominal_angles_deg": [0, 45, 90, 135, 180, 225, 270, 315],
  "num_frames": 8,
  "frames": [
    {
      "stop_id": 0,
      "color_file": "stop_00/color.png",
      "depth_file": "stop_00/depth.npy",
      "timestamp": 1731234567.890,
      "nominal_angle_deg": 0.0,
      "burst_frames_used": 32,
      "quality_report": { "motion_mm": 3.1, "valid_depth_frac": 0.87, "is_flagged": false },
      "accept_reject_history": ["ACCEPTED"]
    },
    ...
  ]
}
```

`stop_meta.json` per stop:
```json
{
  "stop_id": 0,
  "nominal_angle_deg": 0.0,
  "burst_frame_count": 32,
  "burst_motion_mm": 3.1,
  "valid_depth_frac": 0.87,
  "boundary_clipping": [],
  "retake_count": 0,
  "accept_reject_history": ["ACCEPTED"],
  "landmarks_2d": { "left_shoulder": [120.4, 300.2], ... },
  "timestamps": [1731234567.001, 1731234567.034, ...]
}
```

### Backward compatibility

- **Layout 1**: `meta.json` at root with `frame_XX_color.png` files → `_load_meta_json_session()` unchanged. Missing `turn_direction` / `camera_portrait_rotation` root keys fall back to `cfg.turn_direction` / `cfg.camera_portrait_rotation` as today [`io.py:107-108`].
- **Layout 2**: `stop_XX/` subdirs with `*color*.png` and `*depth*.npy` → `_load_subdirs_session()` unchanged; loads only the first file per stop.
- **Stage 27 layout** is a superset of Layout 2 and adds `meta.json` at root. The loader needs a small extension: when `meta.json` exists **and** `stop_XX/` subdirs exist, prefer the structured burst data.

### How `io.py` feeds burst to `burst.py`

Currently `io.py` loads **one** color+depth array per stop into `StopData.color` and `StopData.depth` [`io.py:140-149`]. Burst frames go into `StopData.raw_burst_colors` and `StopData.raw_burst_depths` (fields exist in `StopData` [`types.py:106-107`]) but `io.py` never populates them.

**Required extension**: In `_load_subdirs_session()`, after loading `color.png` / `depth.npy`, glob all `burst_depth_NNN.npy` files and load them into `StopData.raw_burst_depths`; similarly for `burst_color_NNN.png` → `StopData.raw_burst_colors`. Then call `process_stop_burst(raw_burst_colors, raw_burst_depths, cfg)` to produce `filtered_depth`, `depth_std`, `valid_counts` for the stop.

---

## Q5 — CONTROLLER STATE MACHINE

### States

```
WAIT_FOR_SUBJECT → POSITION_CHECK → TURN_PROMPT → WAIT_STILL →
CAPTURE_BURST → EVALUATE → ACCEPT | RETAKE → (loop or DONE) → ABORT
```

### Detailed state descriptions

| State | Entry condition | Action | Transition | Thresholds |
|---|---|---|---|---|
| **WAIT_FOR_SUBJECT** | Controller start | Stream frames; display live RGB + depth. | → `POSITION_CHECK` when depth valid fraction > threshold **for N frames** | `depth_valid_frac_threshold = 0.30` (existing: `capture_session.py:364`). **N = 5 frames** (NEW). |
| **POSITION_CHECK** | Valid subject detected | Check: (a) no boundary clipping, (b) depth range 0.9–2.0 m (body in frame), (c) head and feet visible, (d) silhouette area > min. | → `TURN_PROMPT` if all pass; → `WAIT_FOR_SUBJECT` if fails for > timeout_s. | (a) `evaluate_pre_save()` border logic [`capture_session.py:380-408`] (b) `depth_max_m` check NEW, (c) NEW semantic check, (d) NEW minimum area check. |
| **TURN_PROMPT** | POSITION_CHECK passed | Display: "Stop `{stop_id}` of 8 — ready. Rotate subject to position `{nominal_angle_deg}°`. Press SPACE or wait." | → `WAIT_STILL` on operator keypress or auto-proceed after `turn_prompt_timeout_s`. | `turn_prompt_timeout_s = 10.0` s (NEW/UNTESTED). |
| **WAIT_STILL** | After TURN_PROMPT | Accumulate rolling window of depth frames; compute centroid displacement. Display stillness bar. | → `CAPTURE_BURST` when `centroid_displacement_mm < stillness_threshold_mm` for `stillness_frames` consecutive frames. | `stillness_threshold_mm = 10.0` (existing: `capture_motion.py:33`). `stillness_frames = 15` (NEW; at 30 fps = 0.5 s). |
| **CAPTURE_BURST** | WAIT_STILL passed | Collect `burst_frame_count` frames by calling `cam.read()` or `cam.read_with_raw_and_colorized()`. Show progress bar. | → `EVALUATE` after `burst_frame_count` frames captured or `burst_timeout_s` elapsed. | `burst_frame_count = 30` (existing: `FullBodyConfig.burst_frame_count` [`fullbody/config.py:29`]). `burst_timeout_s = 5.0` s (NEW). |
| **EVALUATE** | Burst complete | Run `evaluate_burst_stillness()` [`capture_motion.py:140-201`] and `evaluate_pre_save()` [`capture_session.py:279-451`]. Display quality summary. | → `ACCEPT` if no FAIL-level diagnostics and motion not flagged; → `RETAKE` otherwise. | `motion_flagged → RETAKE`. FAIL diagnostics in `evaluate_pre_save()` trigger `RETAKE`. |
| **ACCEPT** | EVALUATE passed | Save burst frames + processed depth to `stop_NN/` subdirectory. Write `stop_meta.json`. Increment stop counter. | → `TURN_PROMPT` if more stops remain; → `DONE` if all 8 stops accepted. | — |
| **RETAKE** | EVALUATE failed | Display reason. Increment retake counter. | → `TURN_PROMPT` for same stop if `retake_count < max_retake_count`; → `ABORT` if exceeded. | `max_retake_count = 3` (NEW/UNTESTED). |
| **DONE** | All 8 stops accepted | Write `meta.json` and `trial_meta.json` with `trial_status: "completed"`. Print summary. | — | — |
| **ABORT** | Max retakes exceeded, fatal error, or operator 'Q' press | Write `trial_meta.json` with `trial_status: "aborted"`, `failure_stage`, `failure_reason`. | — | — |

### Thresholds — existing vs. new/untested

| Threshold | Value | Source |
|---|---|---|
| `depth_valid_frac_threshold` | 0.30 | Existing: `capture_session.py:364` |
| `stillness_threshold_mm` | 10.0 mm | Existing: `capture_motion.py:33` |
| `burst_frame_count` | 30 | Existing: `fullbody/config.py:29` |
| `burst_min_valid_frac` | 0.50 | Existing: `fullbody/config.py:36` |
| `stillness_frames` | 15 consecutive frames | **NEW / UNTESTED** |
| `max_retake_count` | 3 | **NEW / UNTESTED** |
| `turn_prompt_timeout_s` | 10.0 s | **NEW / UNTESTED** |
| `burst_timeout_s` | 5.0 s | **NEW / UNTESTED** |
| `position_check_timeout_s` | 15.0 s | **NEW / UNTESTED** |
| Subject min coverage area | TBD | **NEW / UNTESTED** |

### Operator display at each state

```
╔══════════════════════════════════════════════════════╗
║  Full-Body Scan Controller  [Stop 3 of 8 — 135°]     ║
║  State: WAIT_STILL                                    ║
║  Motion: ██░░░░░░░░  3.2 mm / 10.0 mm threshold      ║
║  Valid depth: 87.4%   |  Boundary: OK                 ║
║  Press: [SPACE] capture  |  [R] retake  |  [Q] abort  ║
╚══════════════════════════════════════════════════════╝
```

---

## Q6 — TESTABILITY WITHOUT HARDWARE

### `FakeCamera` interface

The controller must accept any object implementing these `RealSenseCamera` methods:

```python
class FakeCamera:
    depth_scale: float          # e.g. 0.001
    intrinsics_o3d: o3d.camera.PinholeCameraIntrinsic

    def start(self) -> "FakeCamera": ...
    def stop(self) -> None: ...
    def __enter__(self) -> "FakeCamera": ...
    def __exit__(self, *_) -> None: ...

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (color_bgr HxWx3 uint8, depth_m HxW float32) or (None, None)."""
        ...

    def read_with_raw_and_colorized(self) -> tuple[...]:
        """Return (color_bgr, filtered_depth_m, raw_depth_m, raw_z16, depth_color_bgr, frame_meta)."""
        ...

    def get_device_info(self) -> dict: ...
```

Minimum needed for the controller: `start()`, `stop()`, `read()`, `depth_scale`, `intrinsics_o3d`.  
`read_with_raw_and_colorized()` is needed if the controller saves raw burst frames.

### Using `SyntheticHumanoidCaptureRenderer` from `tests/fullbody/synth_humanoid.py`

The synthetic harness renders a rotating humanoid body from a fixed portrait camera at configurable stops. Its per-stop output (color + depth arrays with realistic sensor noise, dropouts, and flying pixels) can be replayed by a `FakeCamera` as follows:

```python
from tests.fullbody.synth_humanoid import SyntheticHumanoid, render_stop

class SyntheticFakeCamera:
    """FakeCamera backed by pre-rendered synth stop frames."""

    def __init__(self, humanoid, stop_yaw_deg, n_burst=30, cfg=None):
        self._frames = []
        for _ in range(n_burst):
            color, depth, _ = render_stop(humanoid, stop_yaw_deg, cfg)
            self._frames.append((color, depth))
        self._idx = 0
        self.depth_scale = 0.001
        self.intrinsics_o3d = humanoid.intrinsics_o3d

    def start(self): return self
    def stop(self): pass

    def read(self):
        if self._idx >= len(self._frames):
            return None, None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame
```

### Proposed test fixture

```python
@pytest.mark.slow
def test_controller_drives_8_stops(tmp_path):
    """State machine drives FakeCamera through all 8 stops; asserts all transitions."""
    from tests.fullbody.synth_humanoid import SyntheticHumanoid
    from realsense_measure.fullbody.capture_controller import CaptureController, ControllerState

    humanoid = SyntheticHumanoid()
    controller = CaptureController(
        camera_factory=lambda angle_deg: SyntheticFakeCamera(humanoid, angle_deg),
        output_dir=tmp_path / "capture",
        cfg=FullBodyConfig(num_stops=8),
    )

    final_state = controller.run()
    assert final_state == ControllerState.DONE
    assert len(list((tmp_path / "capture").glob("stop_*/depth.npy"))) == 8
    assert (tmp_path / "capture" / "meta.json").exists()
    assert (tmp_path / "capture" / "trial_meta.json").exists()
    meta = json.loads((tmp_path / "capture" / "meta.json").read_text())
    assert meta["trial_status"] == "completed"  # (after Stage 27 adds this to meta.json)
```

---

## Q7 — RISKS AND UNKNOWNS

| Risk / Unknown | Label | Detail |
|---|---|---|
| **`rotate_intrinsics()` correctness** | **Unverified** | `realsense_capture.py:266` calls `rotate_intrinsics()` from `realsense_measure.camera.rotation`. That module was not read. The critical question is whether `cx`/`cy` and `fx`/`fy` are swapped correctly for 90° CW vs CCW. Incorrect swap will silently misplace all 3D points. |
| **Decimation intrinsic consistency** | **Unverified** | `realsense_capture.py:488-492` resizes the decimated depth map back to color resolution using `cv2.INTER_NEAREST`. The intrinsics (`intrinsics_o3d`) are built from the colour stream (native resolution), not from the post-decimation depth stream. This is consistent after the nearest-neighbour upsampling, but the sub-pixel accuracy is degraded. The exact distortion this causes on the fused mesh at 5 mm voxels is **unverified with hardware**. |
| **Temporal filter state across stops** | **Unverified** | `TemporalFilter` maintains internal state across calls. When the subject rotates to a new stop, the temporal filter's history from the previous stop will bleed into the first few frames of the new stop. How many frames of "warm-up" are needed before the temporal filter converges is **unverified**. Mitigation: discard the first 5–10 frames of each burst. |
| **`turn_direction` and `camera_portrait_rotation` not persisted in `meta.json`** | Confirmed gap | `save_raw_capture()` writes `camera_config.rotate_90` to `meta.json` but does **not** write `turn_direction` or `camera_portrait_rotation` at the root level. `io.py` falls back to `cfg` defaults when these keys are absent. If the config differs from defaults, replay will produce wrong nominal angles. |
| **`stillness_threshold_mm = 10 mm` for full body** | **Unverified** | The 10 mm value was designed for head-only captures (smaller ROI). A person's body sway during standing is typically 5–15 mm. At 10 mm, some natural sway may always flag. Empirical calibration on live subjects is needed. |
| **`max_retake_count = 3`** | **Untested** | Arbitrary. Too low a limit could leave the session aborted for marginal hardware/environment conditions; too high wastes operator time. |
| **SDK filter order on real hardware** | **Unverified** | The filter order in `_filter_depth()` matches Intel's recommended RS-SDK post-processing order, but the actual SDK behaviour on firmware ≥ 5.15 with D455f-specific hardware may differ. Must be verified with real device. |
| **Rotation of `depth_color_bgr` in `read_with_raw_and_colorized()`** | Possible bug | `realsense_capture.py:756`: `_, raw_depth_m = rotate_rgbd_arrays(color_bgr, raw_depth_m, ...)` — the first argument is `color_bgr` **after** it was already rotated on line 755. This discards the rotated `color_bgr` return value and silently re-uses a stale reference. Needs code review. |
| **Subject height clipping at 1.3 m with 0.90 m camera** | Confirmed known | `FULLBODY.md:3` documents the crown tangent horizon at Y ≈ 1.733 m. Subjects taller than this will have crown clipped. Stage 27 must check this during POSITION_CHECK. |
| **`_load_subdirs_session()` guesses intrinsics** | Confirmed gap | `io.py:180-183` uses `fx = fy = max(h, w)` with no reference to recorded calibration. This is only valid as an approximation. Stage 27 must write `meta.json` at root with real intrinsics so Layout 1 loader is used instead. |

---

## Minimal Code Change List (Ordered for Independent Implementation)

The following changes are required for Stage 27. **None are implemented here.**

| # | File | Class / Function | Purpose |
|---|---|---|---|
| 1 | `realsense_measure/fullbody/capture_controller.py` (NEW) | `CaptureController`, `ControllerState` | The main state machine: `WAIT_FOR_SUBJECT → POSITION_CHECK → TURN_PROMPT → WAIT_STILL → CAPTURE_BURST → EVALUATE → ACCEPT/RETAKE → DONE/ABORT`. Accepts a camera factory, `FullBodyConfig`, and output dir. |
| 2 | `realsense_measure/fullbody/capture_controller.py` | `FakeCamera` (protocol) | Formal `Protocol` class matching the `RealSenseCamera` interface: `start()`, `stop()`, `read()`, `read_with_raw_and_colorized()`, `depth_scale`, `intrinsics_o3d`. |
| 3 | `realsense_measure/camera/recording.py` | `save_raw_capture()` | Add `turn_direction` and `camera_portrait_rotation` to the root-level `meta.json` document (read from `FullBodyConfig` or a passed-in kwargs dict). |
| 4 | `realsense_measure/fullbody/io.py` | `_load_subdirs_session()` | When `burst_depth_NNN.npy` files exist inside a `stop_XX/` dir, load all of them into `StopData.raw_burst_depths` and all `burst_color_NNN.png` into `StopData.raw_burst_colors`. Call `process_stop_burst()` to produce `filtered_depth`, `depth_std`, `valid_counts`. |
| 5 | `realsense_measure/fullbody/io.py` | `load_capture_session()` | Add "Layout 3" branch: if `meta.json` exists AND `stop_XX/` subdirs exist, prefer to use `_load_meta_json_session()` for stop ordering + intrinsics, then augment each stop with burst frames from the subdir. |
| 6 | `realsense_measure/fullbody/config.py` | `FullBodyConfig` | Add controller fields: `stillness_frames: int = 15`, `max_retake_count: int = 3`, `turn_prompt_timeout_s: float = 10.0`, `burst_timeout_s: float = 5.0`, `position_check_timeout_s: float = 15.0`. |
| 7 | `realsense_measure/camera/rotation.py` | `rotate_intrinsics()` | Audit and add a unit test: verify that for a 848×480 → 480×848 CW rotation, `cx_new = old_cy`, `cy_new = W - 1 - old_cx`, and `fx_new = old_fy`, `fy_new = old_fx`. |
| 8 | `tests/fullbody/test_capture_controller.py` (NEW) | `test_controller_drives_8_stops()` | Hardware-free test: `SyntheticFakeCamera` backed by `synth_humanoid.py`; drives all 8 stops; asserts `DONE` state, correct output directory tree, and `trial_meta.json`. |
| 9 | `realsense_measure/fullbody/cli.py` | New `capture` subcommand | Adds `python -m realsense_measure.fullbody capture --out <dir> [--stops 8] [--dry-run]` wiring to `CaptureController`. |
| 10 | `docs/FULLBODY.md` | Section 3 | Add Stage 27 controller reference and burst storage layout documentation. |

---

*Document ends. Total: 7 questions answered, all file:line references verified from source. Items labeled **Unverified** cannot be confirmed without hardware or reading additional files.*

