import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d

stage4 = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")
pts = np.asarray(stage4.points)

# Find nose tip (minimum Z in face region)
face_mask = (pts[:, 1] >= -0.08) & (pts[:, 1] <= 0.02) & (pts[:, 0] >= 0.0) & (pts[:, 0] <= 0.15)
face_pts = pts[face_mask]
nose_idx = np.argmin(face_pts[:, 2])
nose_tip = face_pts[nose_idx]
print(f"Nose tip (approx): X={nose_tip[0]:+.3f}, Y={nose_tip[1]:+.3f}, Z={nose_tip[2]:+.3f}")

# Relative to nose tip:
# If person faces camera:
# Camera looks in +Z direction.
# So face is at Z ~ 0.42 m. Back of head would be at larger Z (> 0.42 m).
# Nose tip is at Z = 0.421 m.
# Nose tip X is ~ +0.07 m.
print(f"Nose tip X: {nose_tip[0]:+.3f} m")

# From nose tip (+0.07 m):
# Towards negative X: reaches -0.036 m (distance = 0.07 - (-0.036) = 0.106 m = 10.6 cm)
# Towards positive X: reaches +0.180 m (distance = 0.180 - 0.07 = 0.110 m = 11.0 cm)
# That is ~10-11 cm on each side of the nose in the frontal plane!
# But what about the sides of the head (ears/temples) at depth Z > 0.45 m?

print("\n--- Point count distribution in Z depth behind nose (at eye/ear level Y in [-0.08, 0.02]) ---")
level_mask = (pts[:, 1] >= -0.08) & (pts[:, 1] <= 0.02)
level_pts = pts[level_mask]

# Divide into left (X < nose_X) and right (X > nose_X)
left_cam = level_pts[level_pts[:, 0] < nose_tip[0]]
right_cam = level_pts[level_pts[:, 0] >= nose_tip[0]]

print(f"Points on Camera-Left (X < {nose_tip[0]:.2f}): {len(left_cam)}")
print(f"  X range: [{left_cam[:, 0].min():+.3f}, {left_cam[:, 0].max():+.3f}]")
print(f"  Z range: [{left_cam[:, 2].min():+.3f}, {left_cam[:, 2].max():+.3f}]")

print(f"Points on Camera-Right (X >= {nose_tip[0]:.2f}): {len(right_cam)}")
print(f"  X range: [{right_cam[:, 0].min():+.3f}, {right_cam[:, 0].max():+.3f}]")
print(f"  Z range: [{right_cam[:, 2].min():+.3f}, {right_cam[:, 2].max():+.3f}]")

