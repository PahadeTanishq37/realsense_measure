import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
import numpy as np
import open3d as o3d
from config import RegistrationConfig
from registration import compute_rotation_deg, compute_translation_mm

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

world_poses = {}
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

accepted_edges = [
    (5, 0), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10), (12, 11),
    (13, 12), (14, 13), (15, 14), (16, 15), (17, 16), (18, 17), (19, 18),
    (20, 19), (21, 20), (22, 21)
]

# Primary edge data logged during run
edge_info = {
    (5, 0): {"tier": "FPFH Fallback", "fit_src": 0.5338, "mut_fit": 0.7800, "rmse": 3.99, "rot_pw": 19.44, "trans_pw": 140.03, "lm": False, "col": False, "fpfh": True, "res_lm": None},
    (6, 5): {"tier": "Landmark Guided", "fit_src": 0.94, "mut_fit": 0.97, "rmse": 5.9, "rot_pw": 3.1, "trans_pw": 43.0, "lm": True, "col": False, "fpfh": False, "res_lm": 5.9},
    (7, 6): {"tier": "Landmark Guided", "fit_src": 0.99, "mut_fit": 0.99, "rmse": 5.8, "rot_pw": 4.2, "trans_pw": 40.0, "lm": True, "col": False, "fpfh": False, "res_lm": 5.8},
    (8, 7): {"tier": "Landmark Guided", "fit_src": 0.99, "mut_fit": 0.99, "rmse": 3.5, "rot_pw": 2.2, "trans_pw": 29.0, "lm": True, "col": False, "fpfh": False, "res_lm": 3.5},
    (9, 8): {"tier": "Landmark Guided", "fit_src": 0.98, "mut_fit": 0.99, "rmse": 3.0, "rot_pw": 2.2, "trans_pw": 27.0, "lm": True, "col": False, "fpfh": False, "res_lm": 3.0},
    (10, 9): {"tier": "Landmark Guided", "fit_src": 0.99, "mut_fit": 1.00, "rmse": 1.5, "rot_pw": 2.3, "trans_pw": 35.0, "lm": True, "col": False, "fpfh": False, "res_lm": 1.5},
    (11, 10): {"tier": "Landmark Guided", "fit_src": 1.00, "mut_fit": 1.00, "rmse": 1.8, "rot_pw": 3.5, "trans_pw": 55.0, "lm": True, "col": False, "fpfh": False, "res_lm": 1.8},
    (12, 11): {"tier": "Landmark Guided", "fit_src": 0.94, "mut_fit": 0.97, "rmse": 4.4, "rot_pw": 2.7, "trans_pw": 65.0, "lm": True, "col": False, "fpfh": False, "res_lm": 4.4},
    (13, 12): {"tier": "Landmark Guided", "fit_src": 0.86, "mut_fit": 0.94, "rmse": 2.1, "rot_pw": 12.3, "trans_pw": 73.0, "lm": True, "col": False, "fpfh": False, "res_lm": 2.1},
    (14, 13): {"tier": "Landmark Guided", "fit_src": 0.91, "mut_fit": 0.95, "rmse": 2.9, "rot_pw": 12.1, "trans_pw": 94.0, "lm": True, "col": False, "fpfh": False, "res_lm": 2.9},
    (15, 14): {"tier": "Landmark Guided", "fit_src": 0.90, "mut_fit": 1.00, "rmse": 2.0, "rot_pw": 8.3, "trans_pw": 52.0, "lm": True, "col": False, "fpfh": False, "res_lm": 2.0},
    (16, 15): {"tier": "Landmark Guided", "fit_src": 0.97, "mut_fit": 0.99, "rmse": 1.8, "rot_pw": 7.2, "trans_pw": 52.0, "lm": True, "col": False, "fpfh": False, "res_lm": 1.8},
    (17, 16): {"tier": "Landmark Guided", "fit_src": 0.99, "mut_fit": 0.99, "rmse": 1.9, "rot_pw": 3.3, "trans_pw": 42.0, "lm": True, "col": False, "fpfh": False, "res_lm": 1.9},
    (18, 17): {"tier": "Landmark Guided", "fit_src": 0.97, "mut_fit": 0.97, "rmse": 4.3, "rot_pw": 7.1, "trans_pw": 41.0, "lm": True, "col": False, "fpfh": False, "res_lm": 4.3},
    (19, 18): {"tier": "Landmark Guided", "fit_src": 1.00, "mut_fit": 1.00, "rmse": 1.9, "rot_pw": 3.6, "trans_pw": 23.0, "lm": True, "col": False, "fpfh": False, "res_lm": 1.9},
    (20, 19): {"tier": "Landmark Guided", "fit_src": 0.99, "mut_fit": 0.99, "rmse": 2.7, "rot_pw": 2.2, "trans_pw": 19.0, "lm": True, "col": False, "fpfh": False, "res_lm": 2.7},
    (21, 20): {"tier": "Landmark Guided", "fit_src": 0.98, "mut_fit": 0.98, "rmse": 12.5, "rot_pw": 5.9, "trans_pw": 71.0, "lm": True, "col": False, "fpfh": False, "res_lm": 12.5},
    (22, 21): {"tier": "Colored ICP", "fit_src": 0.99, "mut_fit": 0.99, "rmse": 3.5, "rot_pw": 11.5, "trans_pw": 58.0, "lm": False, "col": True, "fpfh": False, "res_lm": None},
}

print("=== 1. PRIMARY REGISTRATION AUDIT TABLE ===")
for (src, tgt) in accepted_edges:
    info = edge_info[(src, tgt)]
    n_src = len(frames[src].points)
    n_tgt = len(frames[tgt].points)
    T_rel = np.linalg.inv(world_poses[tgt]) @ world_poses[src]
    rot_opt = compute_rotation_deg(T_rel[:3, :3])
    trans_opt = compute_translation_mm(T_rel)
    t = T_rel[:3, 3]
    
    print(f"Edge {src:02d} -> {tgt:02d}:")
    print(f"  Method/Tier: {info['tier']}")
    print(f"  Source Points: {n_src} | Target Points: {n_tgt}")
    print(f"  Source-Normalized Fitness: {info['fit_src']:.4f} | Mutual Fitness: {info['mut_fit']:.4f} | RMSE: {info['rmse']:.2f} mm")
    print(f"  Pairwise Motion: Rot={info['rot_pw']:.2f} deg, Trans={info['trans_pw']:.2f} mm")
    print(f"  Pose-Graph Corrected: Rot={rot_opt:.2f} deg, Trans={trans_opt:.2f} mm (delta_t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]m)")
    print(f"  Landmark Guidance Used: {info['lm']} (residual={info['res_lm']} mm)")
    print(f"  Colored ICP Used: {info['col']}")
    print(f"  FPFH Used: {info['fpfh']}")
    print(f"  Final Accepted Transform T_{src:02d}->{tgt:02d}:\n{np.array2string(T_rel, precision=4, suppress_small=True)}")
    print("-" * 60)

