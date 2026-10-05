import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import copy
import numpy as np
import open3d as o3d
from config import PipelineConfig
from reconstruction import fuse_point_clouds

# Load the saved isolated frames
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

# Reconstruct aligned frames
aligned_frames = []
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
    
    p = copy.deepcopy(frames[i])
    p.transform(T)
    aligned_frames.append(p)

accepted_indices = [0] + list(range(5, 23))
acc_clouds = [aligned_frames[i] for i in accepted_indices]

cfg = PipelineConfig()
print("PreprocessConfig params:")
print("  voxel_size_m:", cfg.preprocess.voxel_size_m)
print("  outlier_neighbors:", cfg.preprocess.outlier_neighbors)
print("  outlier_std_ratio:", cfg.preprocess.outlier_std_ratio)

# Step 1: Concatenation
fused_concat = copy.deepcopy(acc_clouds[0])
for c in acc_clouds[1:]:
    fused_concat += c
pts_concat = len(fused_concat.points)
print(f"Step 1 (Concatenation): {pts_concat} points")

# Step 2: Statistical Outlier Removal
fused_sor, ind = fused_concat.remove_statistical_outlier(
    nb_neighbors=cfg.preprocess.outlier_neighbors,
    std_ratio=cfg.preprocess.outlier_std_ratio,
)
pts_sor = len(fused_sor.points)
pts_sor_removed = pts_concat - pts_sor
print(f"Step 2 (Statistical Outlier Removal): {pts_sor} points (removed {pts_sor_removed} pts, {pts_sor_removed/pts_concat*100:.2f}%)")

# Step 3: Voxel Downsampling (0.004 m = 4 mm)
fused_voxel = fused_sor.voxel_down_sample(voxel_size=cfg.preprocess.voxel_size_m)
pts_voxel = len(fused_voxel.points)
print(f"Step 3 (Voxel Downsampling at {cfg.preprocess.voxel_size_m*1000}mm): {pts_voxel} points")

# Check stage4_output.ply
stage4_saved = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")
pts_stage4_saved = len(stage4_saved.points)
print(f"Saved stage4_output.ply: {pts_stage4_saved} points")
print(f"Matches simulated Step 3: {pts_voxel == pts_stage4_saved}")

