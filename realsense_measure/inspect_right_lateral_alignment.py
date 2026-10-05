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

# Let's inspect the right lateral cranial surface:
# Head height Y in [-0.15, 0.02], X in [0.12, 0.22], Z in [0.48, 0.62]
print("=== RIGHT LATERAL REGION (EAR / TEMPLE / RAMUS) ===")
print("Bounding box of right ear/temple: X in [0.12, 0.22], Y in [-0.15, 0.05], Z in [0.48, 0.65]")

right_pts_by_frame = {}
for i in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
    pts_w = np.asarray(aligned_clouds[i].points)
    mask = (pts_w[:, 0] >= 0.12) & (pts_w[:, 1] >= -0.15) & (pts_w[:, 1] <= 0.05) & (pts_w[:, 2] >= 0.48)
    sub_pts = pts_w[mask]
    right_pts_by_frame[i] = sub_pts
    print(f"Frame {i:02d}: {len(sub_pts)} right lateral points | X: [{sub_pts[:, 0].min():.3f}, {sub_pts[:, 0].max():.3f}] | Z: [{sub_pts[:, 2].min():.3f}, {sub_pts[:, 2].max():.3f}]" if len(sub_pts) > 0 else f"Frame {i:02d}: 0 points")

# Now check pairwise distance on this specific right lateral surface!
print("\n=== DISTANCE BETWEEN CONSECUTIVE RIGHT LATERAL SURFACES ===")
for i in range(15, 23):
    prev = i - 1
    p_curr = right_pts_by_frame[i]
    p_prev = right_pts_by_frame[prev]
    if len(p_curr) > 20 and len(p_prev) > 20:
        tree = cKDTree(p_prev)
        dists, _ = tree.query(p_curr, k=1)
        d_mm = dists * 1000.0
        print(f"F{i:02d} -> F{prev:02d}: median={np.median(d_mm):.2f} mm | 90th={np.percentile(d_mm, 90):.2f} mm | max={np.max(d_mm):.2f} mm | % < 5mm={np.mean(d_mm < 5.0)*100:.1f}%")

