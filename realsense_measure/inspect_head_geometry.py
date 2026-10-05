import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d

# Load stage4_output.ply and stage3_output.ply
stage4 = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")
pts = np.asarray(stage4.points)
colors = np.asarray(stage4.colors)

print("Stage 4 points:", len(pts))
print("Bounds:")
print("  X:", pts[:, 0].min(), "to", pts[:, 0].max())
print("  Y:", pts[:, 1].min(), "to", pts[:, 1].max())
print("  Z:", pts[:, 2].min(), "to", pts[:, 2].max())

# In RealSense Frame 00:
# X is camera right (+X) / camera left (-X)
# Y is camera down (+Y) / camera up (-Y)
# Z is camera forward (depth away from camera)

# Frame 00 was frontal.
# Let's inspect slice by slice along Y (vertical):
# Top of head is most negative Y.
# Chin / neck / torso is positive Y.
y_percentiles = np.percentile(pts[:, 1], [0, 10, 25, 50, 75, 90, 100])
print("\nY percentiles (vertical: top of head to torso):", y_percentiles)

# Let's see point distributions in vertical slices
for y_min, y_max, label in [
    (-0.20, -0.10, "Top of head / Forehead"),
    (-0.10,  0.00, "Eyes / Nose / Upper face"),
    ( 0.00,  0.10, "Mouth / Chin / Jaw"),
    ( 0.10,  0.25, "Neck / Shoulders / Chest"),
]:
    mask = (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    slice_pts = pts[mask]
    if len(slice_pts) > 0:
        x_min, x_max = slice_pts[:, 0].min(), slice_pts[:, 0].max()
        z_min, z_max = slice_pts[:, 2].min(), slice_pts[:, 2].max()
        print(f"\n{label} ({len(slice_pts)} pts):")
        print(f"  X span: [{x_min:+.3f}, {x_max:+.3f}] -> dX = {x_max - x_min:.3f} m ({(x_max - x_min)*1000:.1f} mm)")
        print(f"  Z span: [{z_min:+.3f}, {z_max:+.3f}] -> dZ = {z_max - z_min:.3f} m ({(z_max - z_min)*1000:.1f} mm)")

