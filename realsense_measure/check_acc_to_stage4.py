import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

frames = [o3d.io.read_point_cloud(f"scan_output/frame_{i:02d}_isolated.ply") for i in range(23)]
stage3_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage3_output.ply")
stage4_pcd = o3d.io.read_point_cloud("stage_wise_output/scan_test_61/stage4_output.ply")

pts3 = np.asarray(stage3_pcd.points)
pts4 = np.asarray(stage4_pcd.points)

starts = []
lens = []
curr = 0
for f in frames:
    n = len(f.points)
    starts.append(curr)
    lens.append(n)
    curr += n

tree4 = cKDTree(pts4)
accepted_indices = [0] + list(range(5, 23))

print("=== DISTANCE FROM EACH ACCEPTED FRAME (IN STAGE 3) TO STAGE 4 FUSED ===")
print("Frame | N_pts | Med dist [mm] | 90th dist [mm] | Max dist [mm] | % > 5mm | % > 10mm")
for i in accepted_indices:
    st = starts[i]
    n = lens[i]
    pts_frame = pts3[st:st+n]
    dists, _ = tree4.query(pts_frame, k=1)
    d_mm = dists * 1000.0
    med = np.median(d_mm)
    p90 = np.percentile(d_mm, 90)
    pmax = np.max(d_mm)
    pct_5 = np.mean(d_mm > 5.0) * 100.0
    pct_10 = np.mean(d_mm > 10.0) * 100.0
    print(f"{i:02d}    | {n:5d} | {med:6.2f}        | {p90:6.2f}         | {pmax:6.2f}       | {pct_5:5.2f}%  | {pct_10:5.2f}%")

print("\n=== DISTANCE FROM REJECTED FRAMES (01-04) TO STAGE 4 ===")
for i in range(1, 5):
    st = starts[i]
    n = lens[i]
    pts_frame = pts3[st:st+n]
    dists, _ = tree4.query(pts_frame, k=1)
    d_mm = dists * 1000.0
    print(f"{i:02d}    | {n:5d} | med={np.median(d_mm):6.2f} | 90th={np.percentile(d_mm, 90):6.2f} | max={np.max(d_mm):6.2f} | % > 10mm={np.mean(d_mm > 10.0)*100:5.2f}%")

