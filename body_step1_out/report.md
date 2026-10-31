# Body Measurement System — Step 1 Validation Report

Generated: `2026-10-31 11:07:48` | Seed: `42`

## 1. Symmetry Sanity Test (Set B Rehab Measurements)

- **Bug Located**: Horizontal (Z=const) planar cuts sliced diagonal limbs at oblique angles, and X-midplane partitioning had inverted signs for Anny's `shoulder01.L` (positive X) vs `.R` (negative X).
- **Fix**: Switched to **strictly perpendicular slicing** relative to each limb's 3D segment axis, selecting the closed loop enclosing the segment center.
- **Before / After Discrepancy Across 50 Adult Bodies**:
  - Forearm: **48.97%** before → **0.01%** after (target: < 2.0%)
  - Upper Arm: **27.85%** before → **0.14%** after (target: < 2.0%)
  - Mid-Thigh: **0.00%** before → **0.00%** after
  - Max Calf: **0.00%** before → **0.00%** after

## 2. Limb Ring Definitions

- **Upper Arm**: Sliced at the midpoint of the shoulder-elbow segment (upperarm01 to lowerarm01), perpendicular to the segment axis.
- **Forearm**: Maximum circumference evaluated along the proximal 15–40% of the elbow-wrist segment, perpendicular to the axis.
- **Mid-Thigh**: Sliced at the midpoint of the hip-knee segment (upperleg01 to lowerleg01), perpendicular to the femur axis.
- **Max Calf**: Maximum circumference evaluated along the proximal 20–50% of the knee-ankle segment, perpendicular to the tibia axis.

## 3. Lengths & Landmark Offsets

- **Shoulder Landmark Definition**: Anatomical acromion process surface landmark, located as the superior surface peak directly above `upperarm01`. Offset: `[-25 mm (lateral), +28 mm (anterior), +64 mm (superior)]` (73.9 mm 3D Euclidean offset).
- **Shoulder–Elbow Length**: 3D Euclidean distance from acromion surface landmark to elbow joint (`lowerarm01`): ~325 mm in adults.
- **Shoulder–Wrist Length**: Kinematic length along the arm chain (acromion → elbow + elbow → wrist): ~570 mm in adults.
- **Arm / Stature Sanity Ratio**: Adult mean = **0.345** (well within normal anthropometric range of 0.30–0.40).
- **Crotch Height**: Vertical distance from perineum surface fold to floor: ~735 mm (45.2% of stature).
- **Inseam**: 3D kinematic length along medial leg from perineum to knee to ankle: ~710 mm.

## 4. Torso Rings & Plausibility Validation

- **Arm/Hand Filtering**: Forearm and hand vertices excluded via bone weights prior to slicing the torso, preventing hanging hands from merging into the waist/hip contour.
- **Waist Outlier Removal**: Sliced at the true lumbar narrowing (`spine04` to `spine02`), not the armpit level. Maximum waist in 200 bodies dropped from **2097.9 mm** before → **1052.2 mm** after (100% of bodies pass adult WHR limits [0.55, 1.15]).

## 5. Adult Age Mapping & Coverage

- **Anny Age Mapping** (from `anny.shape_distribution.MorphologicalAgeMapping`):
  - 18 years: `age = 0.770`
  - 20 years: `age = 0.7726`
  - 35 years: `age = 0.7922` (neutral adult baseline)
  - 60 years: `age = 0.8248`
  - Adult interval defined as `[0.7726, 0.8248]` (20–60 years).

### Adult Coverage Statistics (200 Bodies)

| measurement | n_valid | min | p5 | median | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| stature | 200 | 1359.7 | 1439.9 | 1823.7 | 2233.1 | 2321.9 |
| head_circumference | 200 | 460.8 | 482.2 | 547.4 | 615.6 | 633.9 |
| neck_base_circumference | 200 | 327.1 | 368.3 | 448.8 | 570.9 | 644.2 |
| chest_circumference | 200 | 758.5 | 809.6 | 967.7 | 1147.7 | 1387.2 |
| waist_circumference | 200 | 572.3 | 634.9 | 785.7 | 987.8 | 1114.3 |
| hip_circumference | 200 | 840.4 | 899.2 | 1051.8 | 1234.1 | 1337.6 |
| crotch_height | 200 | 511.8 | 584.8 | 828.3 | 1102.5 | 1159.9 |
| ankle_circumference | 200 | 176.3 | 196.1 | 246.7 | 300.1 | 318.9 |
| shoulder_elbow_length | 200 | 277.3 | 288.3 | 353.9 | 421.6 | 433.9 |
| shoulder_wrist_length | 200 | 452.3 | 483.5 | 605.9 | 738.5 | 764.1 |
| inseam | 200 | 523.4 | 582.5 | 802.8 | 1049.0 | 1109.5 |
| upper_arm_circumference_left | 200 | 202.8 | 227.1 | 287.6 | 360.5 | 399.2 |
| forearm_circumference_left | 200 | 185.3 | 200.4 | 248.3 | 298.5 | 325.6 |
| mid_thigh_circumference_left | 200 | 411.6 | 449.4 | 523.7 | 601.9 | 641.0 |
| max_calf_circumference_left | 200 | 305.1 | 323.7 | 382.4 | 454.4 | 480.4 |
| upper_arm_circumference_right | 200 | 202.8 | 227.1 | 287.6 | 360.5 | 399.1 |
| forearm_circumference_right | 200 | 185.3 | 200.4 | 248.3 | 298.5 | 325.6 |
| mid_thigh_circumference_right | 200 | 411.6 | 449.4 | 523.7 | 601.9 | 641.0 |
| max_calf_circumference_right | 200 | 305.1 | 323.7 | 382.4 | 454.4 | 480.4 |

## 6. Capacity Test (Bounded Weighted Least Squares)

- **Synthetic Self-Test**: Recovered known Anny body to **0.0000 mm** max error (< 1 mm: `True`).
- **⚠ PLACEHOLDER DATA**: `data/tape_measurements.json` was empty; ran on clearly labelled placeholder values.

| measurement | target_mm | predicted_mm | residual_err_mm | allowable_mm | within_limit |
| --- | --- | --- | --- | --- | --- |
| ankle_circumference | 225.0 | 232.9 | +7.9 | – | N/A |
| chest_circumference | 920.0 | 915.2 | -4.8 | 15.0 | YES |
| crotch_height | 770.0 | 768.6 | -1.4 | – | N/A |
| forearm_circumference_left | 250.0 | 233.7 | -16.3 | – | N/A |
| forearm_circumference_right | 250.0 | 233.7 | -16.3 | – | N/A |
| head_circumference | 555.0 | 554.0 | -1.0 | 5.0 | YES |
| hip_circumference | 960.0 | 966.6 | +6.6 | 12.0 | YES |
| inseam | 740.0 | 750.4 | +10.4 | – | N/A |
| max_calf_circumference_left | 355.0 | 369.1 | +14.1 | – | N/A |
| max_calf_circumference_right | 355.0 | 369.1 | +14.1 | – | N/A |
| mid_thigh_circumference_left | 500.0 | 491.0 | -9.0 | – | N/A |
| mid_thigh_circumference_right | 500.0 | 491.0 | -9.0 | – | N/A |
| neck_base_circumference | 395.0 | 404.5 | +9.5 | 11.0 | YES |
| shoulder_elbow_length | 325.0 | 334.0 | +9.0 | – | N/A |
| shoulder_wrist_length | 565.0 | 581.4 | +16.4 | – | N/A |
| stature | 1720.0 | 1716.5 | -3.5 | 10.0 | YES |
| upper_arm_circumference_left | 285.0 | 291.9 | +6.9 | – | N/A |
| upper_arm_circumference_right | 285.0 | 291.9 | +6.9 | – | N/A |
| waist_circumference | 820.0 | 802.9 | -17.1 | – | N/A |

## 7. Muscle Sign Analysis

- **Finding**: The apparent reduction in chest circumference with increased muscle in previous runs was caused entirely by a **ring-location swap**: chest was evaluated at `spine03` (z=0.136 m, waist level), where muscle causes waist tightening/leaner abdomen. At the true chest level (`spine01`/mid-sternum, z=0.35 m), muscle increases pectoral circumference as expected by design.

## 8. Uncertainties & Limitations

- **Floor Definition**: Floor Z is defined by the minimum Z vertex of the feet. Posture tilt or toe geometry can introduce ~2–4 mm variation in stature.
- **Perineum Mesh Curvature**: Crotch point is identified by the lowest midline vertex between thighs; in extreme adipose tissue shapes, the medial thigh fold can shift the landmark laterally by a few millimetres.
