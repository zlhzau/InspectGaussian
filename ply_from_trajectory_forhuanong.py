import os
import argparse
import numpy as np
import open3d as o3d
import csv
from tqdm import tqdm
import tifffile
import cv2 


# 1. Load the trajectory from KeyFrameTrajectory.txt
def load_trajectory(trajectory_file):
    trajectory = {}
    with open(trajectory_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            timestamp = float(parts[0])
            translation = np.array([float(p) for p in parts[1:4]])
            quaternion = np.array([float(p) for p in parts[4:]])
            trajectory[timestamp] = (translation, quaternion)
    return trajectory


# 2. Load the associations from associations.txt
def load_associations(associations_file):
    associations = {}
    with open(associations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            timestamp = float(parts[0])
            rgb_image = os.path.basename(parts[1])  # Extract the file name from the path
            associations[rgb_image] = timestamp
    return associations


# 3. Read and process the depth .tif image
def load_depth_image(depth_image_path):
    depth = tifffile.imread(depth_image_path)  # Load .tif depth image
    
    depth = depth.astype(np.float32)   # Convert millimeters to meters
    
    # 初始化三张深度图
    depth_0_3000 = np.zeros_like(depth)
    depth_3000_6000 = np.zeros_like(depth)
    depth_above_6000 = np.zeros_like(depth)

    # 分割深度图并处理范围
    # 范围 0-3000
    mask_0_3000 = (depth >= 0) & (depth < 5000)
    depth_0_3000[mask_0_3000] = depth[mask_0_3000]

    # 范围 3000-6000
    mask_3000_6000 = (depth >= 5000) & (depth < 8000)
    depth_3000_6000[mask_3000_6000] = depth[mask_3000_6000]

    # 范围 >6000
    mask_above_6000 = (depth >= 8000)
    depth_above_6000[mask_above_6000] = depth[mask_above_6000]
    return [depth_0_3000,depth_3000_6000,depth_above_6000]


# 4. Process RGB-D images and generate point clouds
def process_rgbd_images(color_image, depth_image, intrinsic, voxel_size=[0.1,0.8,1.5]):
    # Load the color and depth images
    color_raw = o3d.io.read_image(color_image)
    depth_image_l = load_depth_image(depth_image)

    depth_raw0 = o3d.geometry.Image(depth_image_l[0])
    # Create an RGB-D image
    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw0, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image0, intrinsic)
    # Downsample the point cloud
    pcd0 = pcd0.voxel_down_sample(voxel_size=voxel_size[0])
    
    depth_raw1 = o3d.geometry.Image(depth_image_l[1])
    # Create an RGB-D image
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw1, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1, intrinsic)
    # Downsample the point cloud
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size[1])

    depth_raw2 = o3d.geometry.Image(depth_image_l[2])
    # Create an RGB-D image
    rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw2, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2, intrinsic)
    # Downsample the point cloud
    
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size[2])

    pcd = pcd0+ pcd1 +pcd2
    return pcd


def process_rgbd_images_detect(color_image, depth_image, intrinsic, detect_image, voxel_size=[0.01,0.2,0.5]):
    # Load the color and depth images
    color_raw = o3d.io.read_image(color_image)
    detect_image = np.asarray(cv2.imread(detect_image, -1),dtype=np.float32)/255
    depth_image_l = load_depth_image(depth_image)
    
    depth_raw0 = o3d.geometry.Image(depth_image_l[0]*detect_image)
    # Create an RGB-D image
    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw0, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image0, intrinsic)
    # Downsample the point cloud
    pcd0 = pcd0.voxel_down_sample(voxel_size=voxel_size[0])
    
    depth_raw1 = o3d.geometry.Image(depth_image_l[1]*detect_image)
    # Create an RGB-D image
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw1, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1, intrinsic)
    # Downsample the point cloud
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size[1])

    depth_raw2 = o3d.geometry.Image(depth_image_l[2]*detect_image)
    # Create an RGB-D image
    rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw2, convert_rgb_to_intensity=False,depth_trunc=3
    )
    # Create a point cloud from the RGB-D image
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2, intrinsic)
    # Downsample the point cloud
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size[2])

    pcd = pcd0+ pcd1 +pcd2
    return pcd


# 5. Transform point cloud based on the pose (translation and quaternion)
def transform_point_cloud(pcd, translation, quaternion):
    # Normalize quaternion if needed
    quaternion = quaternion / np.linalg.norm(quaternion)

    # Create rotation matrix from quaternion
    R = o3d.geometry.get_rotation_matrix_from_quaternion([
        quaternion[3], quaternion[0], quaternion[1], quaternion[2]
    ])

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    # Transform the point cloud
    pcd.transform(T)
    return pcd


# 6. Main function to process each id folder
def process_id_folder(datapath, id_number, associations, trajectory, intrinsic):
    color_dir = os.path.join(datapath, 'idout', str(id_number), 'color')
    depth_dir = os.path.join(datapath, 'idout', str(id_number), 'depth')
    detect_dir = os.path.join(datapath, 'idout', str(id_number), 'mask')

    csv_file = os.path.join(datapath, 'idout', f'{id_number}_image_namelist.csv')

    output_ply = os.path.join(datapath, 'idout', str(id_number), f'{id_number}.ply')

    # Read the CSV file to get the image names
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        image_names = [row[0] for row in reader if row]

    combined_pcd = o3d.geometry.PointCloud()

    # Loop over each image name
    for image_name in tqdm(image_names, desc=f"Processing ID {id_number}"):
        if image_name in associations:
            timestamp = associations[image_name]
            if timestamp in trajectory:
                translation, quaternion = trajectory[timestamp]

                color_image = os.path.join(color_dir, image_name)
                depth_image = os.path.join(depth_dir, image_name.replace('.png', '.tif'))
                detect_image = os.path.join(detect_dir, image_name)

                if os.path.exists(color_image) and os.path.exists(depth_image):
                    # Generate point cloud from the RGB-D images
                    pcd = process_rgbd_images_detect(color_image, depth_image, intrinsic,detect_image)

                    # Transform the point cloud based on the camera pose
                    transformed_pcd = transform_point_cloud(pcd, translation, quaternion)

                    # Add to the combined point cloud
                    combined_pcd += transformed_pcd
                    combined_pcd.voxel_down_sample(voxel_size=0.1)

    # Downsample the point cloud for better performance
    # combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)

    # Save the combined point cloud to a .ply file
    o3d.io.write_point_cloud(output_ply, combined_pcd)

    print(f"Point cloud saved to {output_ply}")


def process_full_dataset(datapath, associations, trajectory, intrinsic):
    color_dir = os.path.join(datapath, 'color')
    depth_dir = os.path.join(datapath, 'depth')
    output_ply = os.path.join(datapath, 'full.ply')

    combined_pcd = o3d.geometry.PointCloud()

    # Loop over each association
    for rgb_image in tqdm(list(associations.keys()), desc="Processing full dataset"):
        timestamp = associations[rgb_image]
        if timestamp in trajectory:
            translation, quaternion = trajectory[timestamp]

            color_image_path = os.path.join(color_dir, rgb_image)
            depth_image_path = os.path.join(depth_dir, rgb_image.replace('.png', '.tif'))

            if os.path.exists(color_image_path) and os.path.exists(depth_image_path):
                # Generate point cloud from the RGB-D images
                pcd = process_rgbd_images(color_image_path, depth_image_path, intrinsic)

                # Transform the point cloud based on the camera pose
                transformed_pcd = transform_point_cloud(pcd, translation, quaternion)

                # Add to the combined point cloud
                combined_pcd += transformed_pcd
                # combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)

    # Downsample the point cloud for better performance
    # combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.001)

    # Save the combined point cloud to a .ply file
    o3d.io.write_point_cloud(output_ply, combined_pcd)

    print(f"Point cloud saved to {output_ply}")


# Main script to process all id folders
def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from RGB-D images and trajectory data.")
    parser.add_argument("-p", "--datapath", required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    datapath = args.datapath
    
    # Step 1: Load trajectory data
    trajectory_file = os.path.join(datapath, 'CameraTrajectory.txt')
    trajectory = load_trajectory(trajectory_file)

    # Step 2: Load associations
    associations_file = os.path.join(datapath, 'associations.txt')
    associations = load_associations(associations_file)

    # Step 3: Define the intrinsic matrix (modify this according to your camera)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720,
        fx=813.19214658, fy=813.62230969,
        cx=651.42157643, cy=345.43288645
    )

    

    # Step 4: Process each id folder
    idout_path = os.path.join(datapath, 'idout')
    id_csv_files = [f for f in os.listdir(idout_path) if f.endswith('_image_namelist.csv')]
    for csv_file in id_csv_files:
        id_number = os.path.splitext(csv_file)[0].split('_')[0]  # Extract id number from file name
        process_id_folder(datapath, id_number, associations, trajectory, intrinsic)

    # process_full_dataset(datapath, associations, trajectory, intrinsic)


if __name__ == "__main__":
    main()