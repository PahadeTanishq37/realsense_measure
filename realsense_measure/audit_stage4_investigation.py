import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import copy
import numpy as np
import open3d as o3d
from config import PipelineConfig
from scipy.spatial import cKDTree

# 1. Load isolated frames and stage3 output
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

# Trace Step 1: Concatenation
fused_concat = copy.deepcopy(acc_clouds[0])
for c in acc_clouds[1:]:
    fused_concat += c

pts_concat = np.asarray(fused_concat.points)

# Trace Step 2: SOR
fused_sor, ind = fused_concat.remove_statistical_outlier(
    nb_neighbors=cfg.preprocess.outlier_neighbors,
    std_ratio=cfg.preprocess.outlier_std_ratio,
)
pts_sor = np.asarray(fused_sor.points)

# Outlier mask
outlier_mask = np.ones(len(pts_concat), dtype=bool)
outlier_mask[ind] = False
pts_removed_sor = pts_concat[outlier_mask]

# Trace Step 3: Voxel downsample
fused_voxel = fused_sor.voxel_down_sample(voxel_size=cfg.preprocess.voxel_size_m)
pts_voxel = np.asarray(fused_voxel.points)

# Load saved Stage 4
stage4_saved = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")
pts_stage4 = np.asarray(stage4_saved.points)

print("=== QUESTION 4: STATISTICAL OUTLIER REMOVAL (SOR) ANALYSIS ===")
print(f"Total points before SOR: {len(pts_concat)}")
print(f"Total points after SOR:  {len(pts_sor)}")
print(f"Total points removed:    {len(pts_removed_sor)} ({len(pts_removed_sor)/len(pts_concat)*100:.2f}%)")

# Where are removed points located?
# Define regions:
# Head: Y in [-0.20, +0.05]
#   - Forehead/crown: Y < -0.10
#   - Face/nose/cheeks: Y in [-0.10, 0.05], X in [0.0, 0.15], Z < 0.48
#   - Right ear / temple: Y in [-0.15, 0.05], X >= 0.12, Z >= 0.48
#   - Left lateral: Y in [-0.15, 0.05], X < 0.0
# Torso / neck / chest: Y > 0.05
regions = {
    "Forehead / crown (Y < -0.10)": pts_concat[:, 1] < -0.10,
    "Frontal face / nose (Y in [-0.10, 0.05], X in [0.0, 0.15], Z < 0.48)": (pts_concat[:, 1] >= -0.10) & (pts_concat[:, 1] <= 0.05) & (pts_concat[:, 0] >= 0.0) & (pts_concat[:, 0] < 0.15) & (pts_concat[:, 2] < 0.48),
    "Right ear / lateral (Y in [-0.15, 0.05], X >= 0.12, Z >= 0.48)": (pts_concat[:, 1] >= -0.15) & (pts_concat[:, 1] <= 0.05) & (pts_concat[:, 0] >= 0.12) & (pts_concat[:, 2] >= 0.48),
    "Left lateral (Y in [-0.15, 0.05], X < 0.0)": (pts_concat[:, 1] >= -0.15) & (pts_concat[:, 1] <= 0.05) & (pts_concat[:, 0] < 0.0),
    "Torso / neck / chest (Y > 0.05)": pts_concat[:, 1] > 0.05,
}

print("\n--- Removal rate by anatomical region in SOR ---")
for r_name, r_mask in regions.items():
    n_total_r = np.sum(r_mask)
    n_removed_r = np.sum(r_mask[outlier_mask])
    pct_removed = (n_removed_r / n_total_r * 100.0) if n_total_r > 0 else 0.0
    print(f"{r_name:65s}: Total={n_total_r:6d} | Removed={n_removed_r:5d} ({pct_removed:5.2f}%)")

print("\n=== QUESTION 5: VOXEL DOWNSAMPLING ANALYSIS (4 mm voxel size) ===")
print(f"Points before voxel downsampling: {len(pts_sor)}")
print(f"Points after voxel downsampling:  {len(pts_voxel)}")
print(f"Compression ratio: {len(pts_sor) / len(pts_voxel):.2f}x reduction ({len(pts_voxel)/len(pts_sor)*100:.2f}% retained)")

for r_name, r_mask in regions.items():
    # Points in SOR
    pts_r_sor = pts_sor[((pts_sor[:, 1] < -0.10) if "Forehead" in r_name else
                         ((pts_sor[:, 1] >= -0.10) & (pts_sor[:, 1] <= 0.05) & (pts_sor[:, 0] >= 0.0) & (pts_sor[:, 0] < 0.15) & (pts_sor[:, 2] < 0.48)) if "Frontal face" in r_name else
                         ((pts_sor[:, 1] >= -0.15) & (pts_sor[:, 1] <= 0.05) & (pts_sor[:, 0] >= 0.12) & (pts_sor[:, 2] >= 0.48)) if "Right ear" in r_name else
                         ((pts_sor[:, 1] >= -0.15) & (pts_sor[:, 1] <= 0.05) & (pts_sor[:, 0] < 0.0)) if "Left lateral" in r_name else
                         (pts_sor[:, 1] > 0.05))]
    
    pts_r_vox = pts_voxel[((pts_voxel[:, 1] < -0.10) if "Forehead" in r_name else
                           ((pts_voxel[:, 1] >= -0.10) & (pts_voxel[:, 1] <= 0.05) & (pts_voxel[:, 0] >= 0.0) & (pts_voxel[:, 0] < 0.15) & (pts_voxel[:, 2] < 0.48)) if "Frontal face" in r_name else
                           ((pts_voxel[:, 1] >= -0.15) & (pts_voxel[:, 1] <= 0.05) & (pts_voxel[:, 0] >= 0.12) & (pts_voxel[:, 2] >= 0.48)) if "Right ear" in r_name else
                           ((pts_voxel[:, 1] >= -0.15) & (pts_voxel[:, 1] <= 0.05) & (pts_voxel[:, 0] < 0.0)) if "Left lateral" in r_name else
                           (pts_voxel[:, 1] > 0.05))]
    print(f"{r_name:65s}: Before={len(pts_r_sor):6d} | After={len(pts_r_vox):5d} ({len(pts_r_vox)/len(pts_r_sor)*100:5.2f}% retained)")

print("\n=== QUESTION 8: QUANTITATIVE COMPARISON STAGE 3 (Accepted) vs STAGE 4 ===")
def stats(name, p):
    c = p.mean(axis=0)
    pmin = p.min(axis=0)
    pmax = p.max(axis=0)
    ext = pmax - pmin
    vol = ext[0] * ext[1] * ext[2]
    density = len(p) / vol if vol > 0 else 0
    print(f"{name}:")
    print(f"  Total points: {len(p)}")
    print(f"  Centroid: [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}] m")
    print(f"  Bounds: X=[{pmin[0]:+.4f}, {pmax[0]:+.4f}], Y=[{pmin[1]:+.4f}, {pmax[1]:+.4f}], Z=[{pmin[2]:+.4f}, {pmax[2]:+.4f}] m")
    print(f"  Extent: dX={ext[0]:.4f} m ({ext[0]*1000:.1f} mm), dY={ext[1]:.4f} m ({ext[1]*1000:.1f} mm), dZ={ext[2]:.4f} m ({ext[2]*1000:.1f} mm)")
    print(f"  Bounding Box Volume: {vol*1000:.2f} liters")
    print(f"  Mean Point Density: {density:.0f} pts/m^3 ({density*1e-6:.2f} pts/cm^3)")

stats("Stage 3 (Concatenated 19 Accepted Frames)", pts_concat)
print()
stats("Stage 4 (Fused Reconstruction)", pts_stage4)

