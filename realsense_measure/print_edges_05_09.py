import sys
sys.path.append(r"c:\Users\pahad\Downloads\realsense_measure\realsense_measure-main\realsense_measure")
from generate_full_audit_tables import edge_info, world_poses, frames
import numpy as np
from registration import compute_rotation_deg, compute_translation_mm

for (src, tgt) in [(5, 0), (6, 5), (7, 6), (8, 7), (9, 8)]:
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
