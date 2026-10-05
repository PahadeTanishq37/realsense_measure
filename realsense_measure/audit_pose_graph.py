import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d
from config import RegistrationConfig
from registration import (
    compute_rotation_deg,
    compute_translation_mm,
    compute_mutual_fitness,
    get_information_matrix,
    optimize_pose_graph,
)

# Load isolated frames
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

world_poses_opt = {}
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
    world_poses_opt[i] = T

accepted_frames = [0] + list(range(5, 23))

# Now let's check odometry chain:
# If there were no loop closures (unoptimized odometry chain from pairwise registration):
# pose_i = pose_{prev} @ T_{prev->i}
# In Stage 3 output, what are the relative transforms between consecutive frames?
print("=== CONSECUTIVE RELATIVE TRANSFORMS IN OPTIMIZED GRAPH ===")
for idx in range(len(accepted_frames) - 1):
    f_prev = accepted_frames[idx]
    f_curr = accepted_frames[idx+1]
    T_prev = world_poses_opt[f_prev]
    T_curr = world_poses_opt[f_curr]
    T_rel = np.linalg.inv(T_prev) @ T_curr
    rot = compute_rotation_deg(T_rel[:3, :3])
    trans = compute_translation_mm(T_rel)
    print(f"Edge {f_curr:02d} -> {f_prev:02d}: rot={rot:5.2f} deg, trans={trans:6.2f} mm")

