import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d

# Load stage3 and stage4
stage3_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage3_output.ply")
stage4_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")

pts3 = np.asarray(stage3_pcd.points)
pts4 = np.asarray(stage4_pcd.points)

# Right ear/lateral mask:
# X in [0.12, 0.20], Y in [-0.12, 0.02], Z in [0.48, 0.62]
mask3 = (pts3[:, 0] >= 0.12) & (pts3[:, 0] <= 0.20) & (pts3[:, 1] >= -0.12) & (pts3[:, 1] <= 0.02) & (pts3[:, 2] >= 0.48) & (pts3[:, 2] <= 0.62)
mask4 = (pts4[:, 0] >= 0.12) & (pts4[:, 0] <= 0.20) & (pts4[:, 1] >= -0.12) & (pts4[:, 1] <= 0.02) & (pts4[:, 2] >= 0.48) & (pts4[:, 2] <= 0.62)

ear3 = pts3[mask3]
ear4 = pts4[mask4]

print(f"Right ear/lateral region:")
print(f"  Stage 3 points: {len(ear3)}")
print(f"  Stage 4 points: {len(ear4)}")
print(f"  Ratio: {len(ear3)/len(ear4):.1f}x fewer points in Stage 4!")
print(f"  Stage 3 bounds X: [{ear3[:,0].min():.3f}, {ear3[:,0].max():.3f}], Y: [{ear3[:,1].min():.3f}, {ear3[:,1].max():.3f}], Z: [{ear3[:,2].min():.3f}, {ear3[:,2].max():.3f}]")
print(f"  Stage 4 bounds X: [{ear4[:,0].min():.3f}, {ear4[:,0].max():.3f}], Y: [{ear4[:,1].min():.3f}, {ear4[:,1].max():.3f}], Z: [{ear4[:,2].min():.3f}, {ear4[:,2].max():.3f}]")

# Compute point spacing (average nearest neighbor distance within ear3 vs ear4)
from scipy.spatial import cKDTree
tree3 = cKDTree(ear3)
d3, _ = tree3.query(ear3, k=2)
spacing3 = np.median(d3[:, 1]) * 1000.0

tree4 = cKDTree(ear4)
d4, _ = tree4.query(ear4, k=2)
spacing4 = np.median(d4[:, 1]) * 1000.0

print(f"  Median point-to-point spacing on ear surface:")
print(f"    Stage 3: {spacing3:.2f} mm")
print(f"    Stage 4: {spacing4:.2f} mm")

