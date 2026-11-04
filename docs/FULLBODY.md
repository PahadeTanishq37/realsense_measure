# Model-Free Full-Body 3D Scanning & Anthropometry Pipeline

## 1. Overview and Architecture

The `realsense_measure.fullbody` package implements a model-free full-body 3D scanning and anthropometry system using a single Intel RealSense D455/D455f camera in portrait orientation.

Unlike parametric model fitting (SMPL, SMPL-X, STAR, or Anny), this pipeline reconstructs the subject directly from multi-stop fused depth observations using volumetric TSDF integration and extracts ISO 7250 / tape-comparable body circumferences and linear dimensions geometrically.

### Licensing & Independence
- **Commercial clean licensing**: Permissive Apache 2.0 / BSD / MIT dependencies only (`open3d`, `scipy`, `numpy`, `opencv-python`, `mediapipe`).
- **No proprietary parametric models**: Zero SMPL, SMPL-X, STAR, or Anny code or weights are included or vendored.
- **Body template**: Strictly an optional interface (`BodyTemplate`) with lazy plug-in loading. All core scan, fusion, and measurement operations work without any template.

---

## 2. Coordinate Frames and Units

- **Units**: Metres ($m$) across all 3D representations; anthropometric measurement results are reported in millimetres ($mm$).
- **Canonical Subject Frame (ISB Convention)**:
  - $+X$: Subject's anatomical left
  - $+Y$: Superior (upwards toward crown)
  - $+Z$: Anterior (forward from chest)
  - **Origin**: Floor plane directly beneath the subject's vertical centre-of-mass axis ($X=0, Y=0, Z=0$).
- **Camera Optical Frame**:
  - $+X$: Right in image plane
  - $+Y$: Downwards in image plane
  - $+Z$: Forward along camera line of sight
  - Converted at the boundary via consensus floor gravity transformation $T_{\text{grav\_cam}}$. Frames are never mixed internally.

---

## 3. Capture Protocol

1. **Hardware Configuration**:
   - Intel RealSense D455 / D455f.
   - Portrait orientation (90° rotated mount).
   - Mounting height: $\approx 0.90\text{ m}$ above the floor, level pitch ($0^\circ$).
   - Subject standing distance: $\approx 1.2 - 1.4\text{ m}$ from the camera lens.
   - Floor must remain clearly visible in the lower 35% of the frame.
2. **Subject Posture**:
   - Standing upright, relaxed normal respiration, looking forward.
   - Fitted, matte clothing (non-reflective, avoiding deep absorbing blacks).
   - Feet slightly separated ($\approx 15\text{ cm}$).
   - Arms in canonical A-pose: abducted $30^\circ - 40^\circ$ from torso, wrists relaxed.
3. **Rotation Protocol**:
   - The subject turns in place through $N$ stops (default 8 stops, $45^\circ$ apart; 4 stops at $90^\circ$ supported).
   - At each stop, subject holds still for a $1.0 - 2.0\text{ s}$ burst ($\approx 30$ frames).

---

## 4. Algorithmic Pipeline

```
   Raw Capture Session (N stops x 30 burst frames)
                        │
                        ▼
            1. Per-Stop Burst Processing
      (Temporal median, sample std, flying-pixel removal)
                        │
                        ▼
              2. Floor Plane RANSAC
     (Consensus gravity frame, locks height + roll + pitch)
                        │
                        ▼
      3. Keypoints & Skeletal Estimation (MediaPipe Pose)
          (Robust local-window 3D lifting with MAD)
                        │
                        ▼
             4. Multi-Stop Alignment
   (Keypoint yaw prior -> Bounded 4-DoF rigid-part ICP -> Loop closure)
                        │
                        ▼
        5. Scalable TSDF Volumetric Fusion
        (Confidence-weighted, hole map calculation)
                        │
                        ▼
       6. Anthropometric Measurement Slicing
   (Trunk arm-masked, perpendicular limb rings, dual polar & convex hull)
```

### Detailed Stages

1. **Burst Processing (`burst.py`)**:
   - Computes per-pixel temporal median of valid depths across the burst stack.
   - Rejects pixels with valid ratio $< 50\%$.
   - Strips flying edge pixels where spatial depth gradient exceeds $4\text{ cm}$.
   - Performs conservative local hole-filling for isolated dropouts ($r \le 3\text{ px}$).
2. **Floor Detection & Gravity Alignment (`floor.py`)**:
   - Runs RANSAC on 3D points back-projected from the bottom 35% of the image.
   - Verifies normal alignment with ceiling vector ($[0, -1, 0]$ in camera optical frame).
   - Computes Rodrigues rotation $R_{\text{grav\_cam}}$ and translation aligning the floor to $Y=0$.
   - Verifies cross-stop floor normal consistency ($< 4^\circ$ discrepancy across all stops).
3. **Confidence Modeling (`confidence.py`)**:
   - Evaluates multi-factor confidence weight $w = w_{\text{angle}} \cdot w_{\text{edge}} \cdot w_{\text{noise}} \cdot w_{\text{temp}} \cdot w_{\text{valid}}$.
   - $w_{\text{angle}} = \cos^2(\theta)$ incidence angle falloff.
   - $w_{\text{edge}}$ Euclidean distance to depth boundaries.
   - $w_{\text{noise}} = 1 / (1 + (\sigma(z) / 5\text{mm})^2)$ with quadratic distance model $\sigma(z) = 1\text{mm} \cdot (z / 0.6)^2$.
   - Discards points below $w < 0.15$ prior to integration.
4. **Body Part Labeling (`parts.py`)**:
   - Computes 3D distance to skeletal segments (head, torso, pelvis, upper arms, forearms, thighs, shanks) via capsule geometry.
   - Masks non-rigid arm segments from global torso ICP and circumference rings.
5. **Multi-Stop Alignment (`alignment.py`)**:
   - **4-DoF optimization**: Only yaw $\psi$ and 2D floor translation $[\Delta X, \Delta Z]$ are optimized; height $Y$, pitch, and roll are strictly constrained by the floor plane.
   - **Bounded recovery**: ICP adjustments are constrained ($|\Delta \psi| \le 30^\circ$, $\|\Delta t\| \le 10\text{ cm}$). If ICP residuals fail to improve, the keypoint prior is retained.
   - **Loop closure**: Distributes residual yaw closure drift across complete $360^\circ$ cycles.
6. **Volumetric Fusion & Hole Map (`fusion.py`)**:
   - Fuses aligned stops into an Open3D `ScalableTSDFVolume` ($5\text{ mm}$ voxel, $20\text{ mm}$ truncation distance).
   - Extracts manifold triangle mesh with connected-component filtering ($> 2\%$ largest cluster).
   - Generates an anatomical hole map evaluating missingness in vulnerable areas (crown, crotch, underarms, feet, back of arms).
7. **Measurements (`measure_scan.py`)**:
   - **Stature**: Vertical distance from floor $Y=0$ to 99.8th percentile crown height. Flags `crown_clipped` if crown touches FOV boundary.
   - **Torso rings**: Slices horizontal slabs ($15\text{ mm}$) at neck base, chest, waist, and hip. Arms are masked out using `parts.py`.
   - **Limb rings**: Slices planes perpendicular to the limb long axis at mid-thigh, mid-upper-arm, forearm, and maximum calf belly.
   - **Dual extraction**:
     - *(a) Polar profile*: 180 angular bins ($2^\circ$), periodic circular interpolation, and noise bias correction $r = \sqrt{\max(0, r^2 - \sigma^2)}$.
     - *(b) Convex hull*: 2D convex hull perimeter representing physical tape tension.
   - Quality classification: `OK` ($\ge 70\%$ coverage), `PARTIAL` ($35-70\%$), or `NOT_MEASURABLE` ($< 35\%$, returns `NaN`).
   - Uncertainty: Point-bootstrap standard deviation in millimetres.

---

## 5. CLI Usage

The package is invoked via standard Python module syntax:

```bash
# 1. Full scan & measurement from raw capture
python -m realsense_measure.fullbody scan --capture realsense_measure/raw_captures/body_01 --out scan_output/fullbody

# 2. Extract measurements from existing scan output
python -m realsense_measure.fullbody measure --scan scan_output/fullbody

# 3. Validate against physical tape measurements
python -m realsense_measure.fullbody validate --scan scan_output/fullbody --tape data/tape_measurements.json

# 4. Evaluate capture setup information score
python -m realsense_measure.fullbody sweet-spot --scan scan_output/fullbody

# 5. Interactive 3D review
python -m realsense_measure.fullbody review --scan scan_output/fullbody
```

---

## 6. Known Physical & Optical Limitations

1. **Clothing & Garments**:
   - Loose, draped, or baggy garments (e.g. skirts, oversized hoodies) obscure anatomical contours and will be measured as part of the body perimeter.
   - Deep matte black fabrics absorb near-IR structured light, causing local depth dropouts.
2. **Hair & Head Wear**:
   - Voluminous or fluffy hair elevates the detected crown level above the true skull vertex.
3. **Occlusions & Self-Shadowing**:
   - Underarms, perineum/crotch, and inner thighs experience optical shadows and grazing incidence angles.
4. **Camera Field of View & Clipping**:
   - At $1.3\text{ m}$ distance in portrait mode, subjects taller than $\approx 1.95\text{ m}$ may experience crown or foot truncation if the camera height is not adjusted.

---

## 7. Plugging In an Articulated Body Template

The pipeline is designed to interface with arbitrary clean rigged templates via the `BodyTemplate` protocol in [`realsense_measure/fullbody/template.py`](file:///C:/Users/pahad/OneDrive/Desktop/realsense_measure/realsense_measure-main/realsense_measure/fullbody/template.py).

```python
from realsense_measure.fullbody.template import GenericRiggedTemplate, TemplateFitter

# Load clean mesh assets (OBJ/GLB + joints + skin weights)
template = GenericRiggedTemplate(
    base_vertices=vertices_np,
    faces=faces_np,
    base_joints=joints_np,
    joint_names=joint_names_list,
)

fitter = TemplateFitter(template, cfg)
fitted_mesh, metrics = fitter.fit(fused_pcd, keypoints, hole_map, aligned_stops)
```

---

## 8. Validation Checklist on Real Data

> [!IMPORTANT]
> **Strict Empirical Integrity Rule**: Synthetic tests check code paths only. Real-world measurement accuracy cannot be claimed without empirical validation against physical tape measurements.

- [ ] **Phase 1: Zero-Validation Baseline** (Current Status):
  - Procedural humanoid unit tests verify algorithms and geometry.
  - Real human accuracy: **Not yet validated on physical human scans with tape data**.
- [ ] **Phase 2: Single-Subject Verification**:
  - Acquire clean 8-stop capture of single subject in minimal fitted clothing.
  - Obtain ground-truth anthropometric tape measurements following ISO 7250 protocol.
  - Run `validate` subcommand and establish empirical bias and MAE.
- [ ] **Phase 3: Multi-Subject Repeatability**:
  - Validate across $\ge 5$ subjects with varying statures and BMIs.
  - Evaluate intra-subject scan-rescan repeatability across multiple independent trials.

---

## 9. Architectural Decisions

1. **Floor-Constrained 4-DoF ICP vs 6-DoF ICP**:
   *Decision*: Lock height ($Y$), pitch, and roll to the consensus floor plane; only optimize yaw and 2D floor translation.
   *Reasoning*: Torso cross-sections are near-cylindrical. Full 6-DoF ICP tends to slide vertically and drift in pitch when observing partial turns.
2. **Arm Masking on Torso Rings**:
   *Decision*: Strictly mask arm points from torso slicing slabs using skeletal capsules.
   *Reasoning*: Natural arm sway between stops can cause hands or forearms to bridge into the chest/waist ring, artificially inflating circumferences by 50–150 mm.
3. **Dual Circumference Output**:
   *Decision*: Output both periodic-interpolated polar radius perimeter and 2D convex hull perimeter.
   *Reasoning*: Physical measuring tapes span concavities under tension (closer to a convex hull), whereas 3D geometric surfaces follow depressions (closer to the polar radius contour). Providing both allows transparent calibration.

---

## 10. Synthetic Verification & Hardware Guards

- **Synthetic Humanoid Verification Suite** (`tests/fullbody/synth_humanoid.py`, `test_e2e.py`, `test_missing_coverage.py`):
  - Procedural humanoid generator with lofted elliptical torso/pelvis (analytic Ramanujan perimeters for chest, waist, and hip), ellipsoid head, and cylindrical limbs in A-pose.
  - Open3D `RaycastingScene` multi-stop renderer from a fixed portrait camera ($H=0.90\text{ m}, D=1.30\text{ m}$) simulating 4 and 8 stops.
  - Injected sensor error model: distance quadratic noise $\sigma(z) = 0.001 \cdot (z/0.6)^2$, 10% dropouts, edge flying pixels, per-stop yaw jitter ($\pm 4^\circ$), translation jitter ($\pm 2\text{ cm}$), and inter-stop arm posture changes ($\pm 3^\circ$).
  - Validates full algorithmic chain: burst median $\to$ floor RANSAC $\to$ confidence weighting $\to$ 3D skeletal lifting $\to$ 4-DoF bounded alignment $\to$ TSDF fusion $\to$ anatomical landmark slicing $\to$ tape comparison API and CLI commands (`scan`, `measure`).
- **Hardware Test Guard (`test_physical_camera_smoke_test`)**:
  - `tests/test_realsense_hardware_smoke_test.py:345` is decorated with `@pytest.mark.skipif(not is_physical_d455_connected(), reason="Physical RealSense camera not detected")`.
  - In CI or headless development environments where no physical Intel RealSense D455 camera is plugged into USB, this test skips intentionally and gracefully. All mock and offline hardware tests (`test_recording_and_load_raw_capture_compatibility`, `test_runner_mocked_camera`, etc.) run and pass.

