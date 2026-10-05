import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

frames = [o3d.io.read_point_cloud(f"scan_output/frame_{i:02d}_isolated.ply") for i in range(23)]
stage3_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage3_output.ply")
pts3 = np.asarray(stage3_pcd.points)

starts = []
lens = []
curr = 0
for f in frames:
    n = len(f.points)
    starts.append(curr)
    lens.append(n)
    curr += n

world_poses = {}
aligned_clouds = {}
for i in range(23):
    pts_iso = np.asarray(frames[i].points)
    n = lens[i]
    st = starts[i]
    pts_w = pts3[st:st+n]
    
    c_local = pts_iso.mean(axis=0)
    c_world = pts_w.mean(axis=0)
    
    H = (pts_iso - c_local).T @ (pts_w - c_world)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_world - R @ c_local
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    world_poses[i] = T
    
    p_trans = o3d.geometry.PointCloud(frames[i])
    p_trans.transform(T)
    aligned_clouds[i] = p_trans

print("=== LATERAL FACIAL COVERAGE IN WORLD COORDINATES (FRAMES 12 TO 22) ===")
print("Nose tip is around X = +0.098 m, Y = -0.054 m, Z = +0.421 m.")
print("Face/Head height slice: Y in [-0.15, +0.05] m (top of head down to chin).")
print("\nFrame | Head Pts | Left Lat (X < 0.0) | Frontal (0.0 <= X < 0.15) | Right Lat (X >= 0.15) | X_min [m] | X_max [m] | Z_max (depth behind nose) [m]")

for i in range(12, 23):
    pts_w = np.asarray(aligned_clouds[i].points)
    # Head only mask: Y in [-0.15, +0.05]
    head_mask = (pts_w[:, 1] >= -0.15) & (pts_w[:, 1] <= 0.05)
    h_pts = pts_w[head_mask]
    
    n_left = np.sum(h_pts[:, 0] < 0.0)
    n_front = np.sum((h_pts[:, 0] >= 0.0) & (h_pts[:, 0] < 0.15))
    n_right = np.sum(h_pts[:, 0] >= 0.15)
    
    x_min = h_pts[:, 0].min() if len(h_pts) > 0 else 0
    x_max = h_pts[:, 0].max() if len(h_pts) > 0 else 0
    z_max = h_pts[:, 2].max() if len(h_pts) > 0 else 0
    
    print(f"F{i:02d}   | {len(h_pts):5d}    | {n_left:5d} ({n_left/len(h_pts)*100:4.1f}%)    | {n_front:5d} ({n_front/len(h_pts)*100:4.1f}%)         | {n_right:5d} ({n_right/len(h_pts)*100:4.1f}%)         | [{x_min:+.3f}, {x_max:+.3f}] | {z_max:+.3f} ({(z_max - 0.421)*1000:4.0f}mm behind nose)")

