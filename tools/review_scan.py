"""
Standalone viewer for reviewing any saved .ply point cloud from a previous
scan, outside of running the full pipeline. Opens the exact same kind of
interactive Open3D window used live during Stage 2-5, with full mouse
rotate/zoom/pan -- NOT a mesh viewer, so this correctly handles files that
are raw colored points with no triangles (which is what every .ply this
project saves actually is).

Usage:
    python review_scan.py scan_output/segmented_object.ply
    python review_scan.py scan_output/fused.ply
    python review_scan.py scan_output_archive/2026-10-03_161145/segmented_object.ply
"""
import sys
import open3d as o3d

def main():
    if len(sys.argv) != 2:
        print("Usage: python review_scan.py <path_to_ply_file>")
        sys.exit(1)

    path = sys.argv[1]
    pcd = o3d.io.read_point_cloud(path)
    print(f"Loaded {path}: {len(pcd.points)} points")

    if len(pcd.points) == 0:
        print("Warning: this file has 0 points -- nothing to display.")
        sys.exit(1)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Reviewing: {path}",
        width=1280,
        height=720,
    )

if __name__ == "__main__":
    main()
