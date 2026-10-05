import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import copy
import numpy as np
import open3d as o3d
from config import PipelineConfig

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
fused_concat = copy.deepcopy(acc_clouds[0])
for c in acc_clouds[1:]:
    fused_concat += c

pts_c = np.asarray(fused_concat.points)

fused_sor, ind = fused_concat.remove_statistical_outlier(
    nb_neighbors=cfg.preprocess.outlier_neighbors,
    std_ratio=cfg.preprocess.outlier_std_ratio,
)
pts_s = np.asarray(fused_sor.points)

outlier_mask = np.ones(len(pts_c), dtype=bool)
outlier_mask[ind] = False

fused_voxel = fused_sor.voxel_down_sample(voxel_size=cfg.preprocess.voxel_size_m)
pts_v = np.asarray(fused_voxel.points)

# Check points in the right ear region with Z > 0.590 m
ear_z59_mask = (pts_c[:, 0] >= 0.12) & (pts_c[:, 0] <= 0.20) & (pts_c[:, 1] >= -0.12) & (pts_c[:, 1] <= 0.02) & (pts_c[:, 2] > 0.590)

print("Points in right ear/temple with Z > 0.590m:")
print("  In concatenated Stage 3:", np.sum(ear_z59_mask))
if np.sum(ear_z59_mask) > 0:
    sub_c = pts_c[ear_z59_mask]
    print(f"  Z range: [{sub_c[:, 2].min():.3f}, {sub_c[:, 2].max():.3f}]")
    removed_by_sor = np.sum(outlier_mask[ear_z59_mask])
    print(f"  Removed by SOR: {removed_by_sor} ({removed_by_sor/np.sum(ear_z59_mask)*100:.1f}%)")

ear_z59_sor = (pts_s[:, 0] >= 0.12) & (pts_s[:, 0] <= 0.20) & (pts_s[:, 1] >= -0.12) & (pts_s[:, 1] <= 0.02) & (pts_s[:, 2] > 0.590)
print("  In post-SOR cloud:", np.sum(ear_z59_sor))
if np.sum(ear_z59_sor) > 0:
    sub_s = pts_s[ear_z59_sor]
    print(f"  SOR Z range: [{sub_s[:, 2].min():.3f}, {sub_s[:, 2].max():.3f}]")

ear_z59_vox = (pts_v[:, 0] >= 0.12) & (pts_v[:, 0] <= 0.20) & (pts_v[:, 1] >= -0.12) & (pts_v[:, 1] <= 0.02) & (pts_v[:, 2] > 0.590)
print("  In post-voxel cloud:", np.sum(ear_z59_vox))
if np.sum(ear_z59_vox) > 0:
    sub_v = pts_v[ear_z59_vox]
    print(f"  Voxel Z range: [{sub_v[:, 2].min():.3f}, {sub_v[:, 2].max():.3f}]")

