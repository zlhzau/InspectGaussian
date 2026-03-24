# 文件名：point3d_bin_generate.py
import open3d as o3d
import struct
import numpy as np
import os
import sys


def read_ply_with_open3d(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    ply_points = []
    for point, color in zip(points, colors):
        x, y, z = point
        r, g, b = (color * 255).astype(int)
        ply_points.append((x, y, z, r, g, b))
    return ply_points


def write_points3d_binary(output_file, ply_points):
    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(ply_points)))
        point3D_id = 1
        for point in ply_points:
            xyz = point[:3]
            rgb = point[3:6]
            error = 0.5
            image_ids = [1]
            point2D_idxs = [1]

            f.write(struct.pack('Q', point3D_id))
            f.write(struct.pack('ddd', *xyz))
            f.write(struct.pack('BBB', *rgb))
            f.write(struct.pack('d', error))
            f.write(struct.pack('Q', len(image_ids)))
            for image_id, point2D_idx in zip(image_ids, point2D_idxs):
                f.write(struct.pack('ii', image_id, point2D_idx))
            point3D_id += 1
    print(f"points3D binary file {output_file} created successfully!")


def main(datapath):
    # ply_file_path = os.path.join(datapath, "full_dp.ply")
    # if os.path.exists(ply_file_path):
    #     print(ply_file_path)
    #     sparse_dir = os.path.join(datapath, "sparse", "0")
    #     os.makedirs(sparse_dir, exist_ok=True)
    #     output_file = os.path.join(sparse_dir, "points3D.bin")
    #     ply_points = read_ply_with_open3d(ply_file_path)
    #     write_points3d_binary(output_file, ply_points)

    idout_dir = os.path.join(datapath, "idout")
    if not os.path.exists(idout_dir):
        print(f"Error: Directory not found {idout_dir}")
        return
    
    for id_dir in os.listdir(idout_dir):
        id_dir_path = os.path.join(idout_dir, id_dir)
        if os.path.isdir(id_dir_path):
            ply_file_path = os.path.join(id_dir_path, f"{id_dir}.ply")
            if os.path.exists(ply_file_path):
                sparse_dir = os.path.join(id_dir_path, "sparse", "0")
                os.makedirs(sparse_dir, exist_ok=True)
                output_file = os.path.join(sparse_dir, "points3D.bin")
                ply_points = read_ply_with_open3d(ply_file_path)
                write_points3d_binary(output_file, ply_points)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != '-p':
        print("Usage: python point3d_bin_generate.py -p datapath")
        sys.exit(1)

    datapath = sys.argv[2]
    main(datapath)