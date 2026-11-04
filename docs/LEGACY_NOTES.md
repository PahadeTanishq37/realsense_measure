# Legacy Notes & Anthropometric Reconstruction Lessons

This document summarizes critical geometric insights, coordinate conventions, and empirical findings discovered during the experimental Stages 1–26 body reconstruction iterations before the codebase cleanup.

---

## 1. Coordinate Frames & Transformations

- **Anny Canonical Frame**:
  - $+Z$: Superior (Up)
  - $+Y$: Anterior (Forward)
  - $+X$: Lateral Right-to-Left (Subject Left $> 0$)
- **SMPL Canonical Frame**:
  - $+Z$: Superior (Up)
  - $-Y$: Anterior (Forward)
  - $+X$: Lateral Right-to-Left (Subject Left $> 0$)
- **Target Project Frame (Stage 22 Standardized ISB Frame)**:
  - $+X$: Subject Lateral (Subject Left $> 0$, Right $< 0$)
  - $+Y$: Anatomical Superior (Inferior $\to$ Superior; Head $> 0$, Feet $< 0$)
  - $+Z$: Anatomical Anterior (Posterior $\to$ Anterior; Chest $> 0$, Back $< 0$)
- **Orthonormal Canonical-to-ISB Rotation Matrix ($R_{\text{ISB}}$)**:
  $$\mathbf{R}_{\text{ISB}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \det(\mathbf{R}_{\text{ISB}}) = +1.0$$
  This pure right-handed rotation maps canonical SMPL $[X_c, Y_c, Z_c]$ to $[X_{\text{isb}}, Y_{\text{isb}}, Z_{\text{isb}}]$ such that $X_{\text{isb}} = X_c$, $Y_{\text{isb}} = Z_c$, and $Z_{\text{isb}} = -Y_c$.

---

## 2. Anthropometric Slicing Geometry

- **Limb Circumferences**:
  - Never slice limbs with global horizontal planes ($Y = \text{const}$).
  - Limbs in A-pose or natural posture tilt at oblique angles. Slicing horizontally yields an exaggerated elliptical cross-section.
  - Slicing planes must be constructed strictly **perpendicular to the local bone/segment longitudinal axis** (e.g., normal aligned with vector from radiale to stylion for forearm, or femur vector for thigh).
- **Torso Ring Extraction & Arm Masking**:
  - Transverse cutting planes through the chest and waist will intersect resting upper arms and forearms unless filtered out.
  - Slicing must isolate the midline trunk polygon (enclosing $X = 0$, filtering out lateral disconnected contours with $|X| > \text{bound}$) or mask out arm vertices before perimeter calculation to prevent massive overestimation (e.g. waist inflated to $1,400+\text{ mm}$).
- **Contour Closure Verification**:
  - Verify that planar intersection polylines form continuous closed loops before reporting perimeters.
  - Require closing gap ratio $\le 1.8 \times$ median segment length and isoperimetric circularity quotient $\frac{4\pi A}{P^2} \ge 0.10$.

---

## 3. Empirical Observations & Full-Body Limitations

- **Turntable / Stop-and-Hold Registration**:
  - RealSense depth registration via unconstrained point-to-plane ICP between 90° yaw stops did not reliably improve coarse turntable yaw angles. Coarse mechanical/protocol angle priors were more stable than ICP on human surfaces prone to non-rigid posture shifts.
- **Surface Coverage**:
  - 4 orthogonal stops (front, back, left profile, right profile) achieve approximately $\sim 71\%$ observed surface coverage around torso cross-sections.
  - Top of head (crown) and lower legs/feet were clipped by the vertical field of view of a single static D455 sensor placed at chest height.
- **Validation Reality**:
  - Algorithmic slicing and geometry pipelines were validated synthetically against analytical cylinders and prisms.
  - Clinical/physical anthropometric accuracy cannot be claimed without calibrating against manual anthropometric ground truth (calipers and tape measurements) on physical human subjects.
