import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d

frames = [o3d.io.read_point_cloud(f"scan_output/frame_{i:02d}_isolated.ply") for i in range(23)]

print("=== STAGE 2 ISOLATED POINT CLOUDS: FRAMES 12 TO 22 ===")
print("Frame | N_pts | Centroid (cam XYZ) [m] | Extent (dX, dY, dZ) [m] | Y_min (top) | Y_max (bot)")
for i in range(12, 23):
    pts = np.asarray(frames[i].points)
    c = pts.mean(axis=0)
    p_min = pts.min(axis=0)
    p_max = pts.max(axis=0)
    ext = p_max - p_min
    # Let's see how many points are in the head (top 25 cm of Y) vs torso (bottom of Y)
    # In camera frame, Y is down.
    head_pts = np.sum(pts[:, 1] < (p_min[1] + 0.25))
    torso_pts = np.sum(pts[:, 1] >= (p_min[1] + 0.25))
    print(f"Frame {i:02d} | {len(pts):5d} | [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}] | [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] | {p_min[1]:+.3f}     | {p_max[1]:+.3f} | Head: {head_pts} ({head_pts/len(pts)*100:.1f}%), Torso: {torso_pts} ({torso_pts/len(pts)*100:.1f}%)")

