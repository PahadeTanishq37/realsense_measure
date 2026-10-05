import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

stage3_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage3_output.ply")
stage4_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")

pts3 = np.asarray(stage3_pcd.points)
pts4 = np.asarray(stage4_pcd.points)

print("Stage 3 pts:", len(pts3))
print("Stage 4 pts:", len(pts4))

# Check distance from every Stage 3 point to nearest Stage 4 point
tree4 = cKDTree(pts4)
dists3_to_4, _ = tree4.query(pts3, k=1)
d_mm = dists3_to_4 * 1000.0

print(f"Distance from Stage 3 to Stage 4:")
print(f"  Median: {np.median(d_mm):.2f} mm")
print(f"  90th percentile: {np.percentile(d_mm, 90):.2f} mm")
print(f"  99th percentile: {np.percentile(d_mm, 99):.2f} mm")
print(f"  Max: {np.max(d_mm):.2f} mm")
print(f"  Points with d > 10 mm: {np.sum(d_mm > 10.0)} ({np.mean(d_mm > 10.0)*100:.2f}%)")
print(f"  Points with d > 20 mm: {np.sum(d_mm > 20.0)} ({np.mean(d_mm > 20.0)*100:.2f}%)")

# Check where the points with d > 10 mm are located
lost_pts = pts3[d_mm > 10.0]
if len(lost_pts) > 0:
    print("Lost points bounds:", lost_pts.min(axis=0), "to", lost_pts.max(axis=0))

