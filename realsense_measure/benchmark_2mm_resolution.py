import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

p3 = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage3_output.ply")
pts3 = np.asarray(p3.points)

pts_acc = np.vstack([pts3[:6255], pts3[45224:]])
p_acc = o3d.geometry.PointCloud()
p_acc.points = o3d.utility.Vector3dVector(pts_acc)

# Benchmark 4mm + SOR (Old human path)
t0 = time.perf_counter()
p_old_sor, _ = p_acc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
p_old_vox = p_old_sor.voxel_down_sample(voxel_size=0.004)
t_old_ms = (time.perf_counter() - t0) * 1000.0

# Benchmark 4mm NO SOR (Previous step)
t0 = time.perf_counter()
p_4mm = p_acc.voxel_down_sample(voxel_size=0.004)
t_4mm_ms = (time.perf_counter() - t0) * 1000.0

# Benchmark 2mm NO SOR (New step)
t0 = time.perf_counter()
p_2mm = p_acc.voxel_down_sample(voxel_size=0.002)
t_2mm_ms = (time.perf_counter() - t0) * 1000.0

pts_old = np.asarray(p_old_vox.points)
pts_4mm = np.asarray(p_4mm.points)
pts_2mm = np.asarray(p_2mm.points)

print("=== RESOLUTION & RUNTIME COMPARISON ===")
print(f"Old Human (4mm + SOR):   {len(pts_old):6d} pts | Runtime: {t_old_ms:6.1f} ms")
print(f"4mm NO SOR:              {len(pts_4mm):6d} pts | Runtime: {t_4mm_ms:6.1f} ms")
print(f"2mm NO SOR:              {len(pts_2mm):6d} pts | Runtime: {t_2mm_ms:6.1f} ms")

# Region breakdowns
regions = {
    "Total Cloud": lambda p: np.ones(len(p), dtype=bool),
    "Forehead / crown (Y < -0.10)": lambda p: p[:, 1] < -0.10,
    "Frontal face / nose (Y in [-0.10, 0.05], X in [0.0, 0.15], Z < 0.48)": lambda p: (p[:, 1] >= -0.10) & (p[:, 1] <= 0.05) & (p[:, 0] >= 0.0) & (p[:, 0] < 0.15) & (p[:, 2] < 0.48),
    "Right ear / lateral (Y in [-0.15, 0.05], X >= 0.12, Z >= 0.48)": lambda p: (p[:, 1] >= -0.15) & (p[:, 1] <= 0.05) & (p[:, 0] >= 0.12) & (p[:, 2] >= 0.48),
    "Extreme profile ear (X in [0.12, 0.20], Y in [-0.12, 0.02], Z > 0.590)": lambda p: (p[:, 0] >= 0.12) & (p[:, 0] <= 0.20) & (p[:, 1] >= -0.12) & (p[:, 1] <= 0.02) & (p[:, 2] > 0.590),
    "Torso / neck / chest (Y > 0.05)": lambda p: p[:, 1] > 0.05,
}

print("\n=== ANATOMICAL REGION POINT COUNTS ===")
print(f"{'Region':<65} | {'Old 4mm+SOR':<11} | {'4mm NO SOR':<10} | {'2mm NO SOR':<10} | {'Gain vs Old':<11}")
for r_name, r_fn in regions.items():
    n_old = np.sum(r_fn(pts_old))
    n_4mm = np.sum(r_fn(pts_4mm))
    n_2mm = np.sum(r_fn(pts_2mm))
    gain = f"{n_2mm / max(n_old, 1):.1f}x"
    print(f"{r_name:<65} | {n_old:<11d} | {n_4mm:<10d} | {n_2mm:<10d} | {gain:<11}")

# Spatial spacing on the right ear surface
mask_ear_old = (pts_old[:, 0] >= 0.12) & (pts_old[:, 0] <= 0.20) & (pts_old[:, 1] >= -0.12) & (pts_old[:, 1] <= 0.02) & (pts_old[:, 2] >= 0.48)
mask_ear_4mm = (pts_4mm[:, 0] >= 0.12) & (pts_4mm[:, 0] <= 0.20) & (pts_4mm[:, 1] >= -0.12) & (pts_4mm[:, 1] <= 0.02) & (pts_4mm[:, 2] >= 0.48)
mask_ear_2mm = (pts_2mm[:, 0] >= 0.12) & (pts_2mm[:, 0] <= 0.20) & (pts_2mm[:, 1] >= -0.12) & (pts_2mm[:, 1] <= 0.02) & (pts_2mm[:, 2] >= 0.48)

tree_old = cKDTree(pts_old[mask_ear_old])
d_old, _ = tree_old.query(pts_old[mask_ear_old], k=2)

tree_4mm = cKDTree(pts_4mm[mask_ear_4mm])
d_4mm, _ = tree_4mm.query(pts_4mm[mask_ear_4mm], k=2)

tree_2mm = cKDTree(pts_2mm[mask_ear_2mm])
d_2mm, _ = tree_2mm.query(pts_2mm[mask_ear_2mm], k=2)

print("\n=== RIGHT EAR SURFACE SPATIAL SPACING ===")
print(f"Old 4mm + SOR:   Median point spacing = {np.median(d_old[:, 1])*1000:.2f} mm | Z_max = {pts_old[mask_ear_old, 2].max():.3f} m")
print(f"4mm NO SOR:      Median point spacing = {np.median(d_4mm[:, 1])*1000:.2f} mm | Z_max = {pts_4mm[mask_ear_4mm, 2].max():.3f} m")
print(f"2mm NO SOR:      Median point spacing = {np.median(d_2mm[:, 1])*1000:.2f} mm | Z_max = {pts_2mm[mask_ear_2mm, 2].max():.3f} m")

# Memory footprint
mem_old_kb = pts_old.nbytes / 1024
mem_4mm_kb = pts_4mm.nbytes / 1024
mem_2mm_kb = pts_2mm.nbytes / 1024
print(f"\nEstimated Point Cloud Array RAM: Old={mem_old_kb:.1f} KB, 4mm={mem_4mm_kb:.1f} KB, 2mm={mem_2mm_kb:.1f} KB (0.002 GB)")

