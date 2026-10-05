import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d

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

accepted_frames = [0] + list(range(5, 23))

print("=== INDIVIDUAL FRAME EXTENTS IN WORLD COORDINATES ===")
print("Frame | N_pts | Centroid (X, Y, Z) [m] | Extent [m] (dX, dY, dZ) | X_min, X_max [m] | Y_min, Y_max [m] | Z_min, Z_max [m]")
for i in accepted_frames:
    pts_w = np.asarray(aligned_clouds[i].points)
    c = pts_w.mean(axis=0)
    p_min = pts_w.min(axis=0)
    p_max = pts_w.max(axis=0)
    extent = p_max - p_min
    print(f"{i:02d}    | {len(pts_w):5d} | [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]  | [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]  | [{p_min[0]:+.3f}, {p_max[0]:+.3f}] | [{p_min[1]:+.3f}, {p_max[1]:+.3f}] | [{p_min[2]:+.3f}, {p_max[2]:+.3f}]")

print("\n=== TOTAL BOUNDING BOX OF ACCEPTED FRAMES IN STAGE 3 ===")
all_acc_pts = np.vstack([np.asarray(aligned_clouds[i].points) for i in accepted_frames])
print("Stage 3 Total Acc Pts:", len(all_acc_pts))
print("Min bounds:", all_acc_pts.min(axis=0))
print("Max bounds:", all_acc_pts.max(axis=0))
print("Extent (dX, dY, dZ):", all_acc_pts.max(axis=0) - all_acc_pts.min(axis=0))

# Check Stage 4
stage4_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")
pts4 = np.asarray(stage4_pcd.points)
print("\n=== STAGE 4 FUSED POINT CLOUD ===")
print("Stage 4 Pts:", len(pts4))
print("Min bounds:", pts4.min(axis=0))
print("Max bounds:", pts4.max(axis=0))
print("Extent (dX, dY, dZ):", pts4.max(axis=0) - pts4.min(axis=0))

