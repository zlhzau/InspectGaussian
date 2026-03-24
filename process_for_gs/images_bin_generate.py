# 文件名：images_bin_generate.py
import csv
import os
import sys
import numpy as np
import struct
from scipy.spatial.transform import Rotation as R
import shutil


def load_valid_rgb_image_names(csv_file):
    valid_names = set()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                valid_names.add(row[0])  # 直接使用文件名
    return valid_names


def load_associations(associations_file):
    associations = {}
    with open(associations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            rgb_image_path = parts[1]  # 获取 RGB 图像路径
            rgb_image = os.path.basename(rgb_image_path)  # 提取文件名（忽略路径）
            rgb_timestamp = np.float64(parts[0])  # 获取 RGB 时间戳
            associations[rgb_image] = rgb_timestamp
    return associations


def load_trajectory(trajectory_file):
    trajectory = {}
    with open(trajectory_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            timestamp = np.float64(parts[0])  # 时间戳
            translation = parts[1:4]  # xyz 位移
            quaternion = parts[4:8]  # 四元数 (qx, qy, qz, qw)
            trajectory[timestamp] = translation + quaternion
    return trajectory


def c2w_to_w2c(translation, quaternion):
    r_c2w = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    R_c2w = r_c2w.as_matrix()
    R_w2c = R_c2w.T
    t_c2w = np.array(translation, dtype=float)
    t_w2c = -R_w2c @ t_c2w
    r_w2c = R.from_matrix(R_w2c)
    q_w2c = r_w2c.as_quat()  # 输出 [qx, qy, qz, qw]
    return t_w2c, [q_w2c[3], q_w2c[0], q_w2c[1], q_w2c[2]]


def generate_new_trajectory(valid_rgb_names, associations, trajectory, output_file):
    with open(output_file, 'w') as f:
        for rgb_name in sorted(valid_rgb_names):
            if rgb_name in associations:
                timestamp = associations[rgb_name]
                if timestamp in trajectory:
                    translation = trajectory[timestamp][:3]
                    quaternion = trajectory[timestamp][3:]
                    w2c_translation, w2c_quaternion = c2w_to_w2c(translation, quaternion)
                    f.write(f"{rgb_name} {' '.join(map(str, w2c_translation))} {' '.join(map(str, w2c_quaternion))}\n")


def generate_full_trajectory(associations, trajectory, output_file):
    with open(output_file, 'w') as f:
        for rgb_name, timestamp in sorted(associations.items(), key=lambda x: x[1]):
            if timestamp in trajectory:
                translation = trajectory[timestamp][:3]
                quaternion = trajectory[timestamp][3:]
                w2c_translation, w2c_quaternion = c2w_to_w2c(translation, quaternion)
                f.write(f"{rgb_name} {' '.join(map(str, w2c_translation))} {' '.join(map(str, w2c_quaternion))}\n")


def write_images_binary(output_file, trajectory_file,test_file):
    images = []
    camera_id = 1
    image_id = 1

    with open(trajectory_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            tvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            qvec = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            xys = []
            point3D_ids = []

            images.append({
                'image_id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'image_name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            })
            image_id += 1
            # camera_id += 1
            
    with open(test_file, 'w') as ftest:
        for i, image in enumerate(images):
            # 每 10 个图像中抽取一个
            if i % 10 == 9:
                ftest.write(image['image_name'] + '\n')

    with open(output_file, 'wb') as fid:
        fid.write(struct.pack('Q', len(images)))
        for image in images:
            fid.write(struct.pack('i', image['image_id']))
            fid.write(struct.pack('dddd', *image['qvec']))
            fid.write(struct.pack('ddd', *image['tvec']))
            fid.write(struct.pack('i', image['camera_id']))
            fid.write(image['image_name'].encode('utf-8'))
            fid.write(b'\x00')
            fid.write(struct.pack('Q', len(image['xys'])))
            for xy, point3D_id in zip(image['xys'], image['point3D_ids']):
                fid.write(struct.pack('ddq', xy[0], xy[1], point3D_id))

    print(f"Binary file {output_file} created successfully!")


def main(datapath):
    associations_file = os.path.join(datapath, "associations.txt")
    trajectory_file = os.path.join(datapath, "KeyFrameTrajectory.txt")
    full_output_file = os.path.join(datapath, "full_CameraTrajectory.txt")

    # generate_full_trajectory(load_associations(associations_file), load_trajectory(trajectory_file), full_output_file)
    # sparse_dir = os.path.join(datapath, "sparse", "0")
    # os.makedirs(sparse_dir, exist_ok=True)
    # write_images_binary(os.path.join(sparse_dir, "images.bin"), full_output_file,os.path.join(sparse_dir, "test.txt"))

    idout_dir = os.path.join(datapath, "idout")
    if os.path.exists(idout_dir):
        for id_dir in os.listdir(idout_dir):
            id_dir_path = os.path.join(idout_dir, id_dir)
            if os.path.isdir(id_dir_path):
                csv_file = os.path.join(idout_dir, f"{id_dir}_image_namelist.csv")
                if os.path.exists(csv_file):
                    valid_rgb_names = load_valid_rgb_image_names(csv_file)
                    output_file = os.path.join(id_dir_path, f"{id_dir}_CameraTrajectory.txt")
                    generate_new_trajectory(valid_rgb_names, load_associations(associations_file), load_trajectory(trajectory_file), output_file)
                    sparse_dir = os.path.join(id_dir_path, "sparse", "0")
                    os.makedirs(sparse_dir, exist_ok=True)
                    write_images_binary(os.path.join(sparse_dir, "images.bin"), output_file,os.path.join(sparse_dir, "test.txt"))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != '-p':
        print("Usage: python images_bin_generate.py -p datapath")
        sys.exit(1)

    datapath = sys.argv[2]
    main(datapath)