import os
import time  # [新增] 用于计时
import torch # [新增] 用于获取GPU信息
from shutil import copyfile
import cv2
import numpy as np
from ultralytics import YOLOWorld
import open3d as o3d
from tqdm import tqdm
from ImageTree import ImageTree, IDBoxTree, IDSET
import ply_from_trajectory_forhuanong
import csv

# -----------------------------
# Helper
# -----------------------------

def draw_boxes_with_labels(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_point_center(points, depth_image, max_distance=3.5):
    # 最左x
    p_left_x = points[0]
    # 最右x
    p_right_x = points[2]
    # 最上方y
    p_top_y = points[1]
    # 最下方y
    p_bottom_y = points[3]
    # 计算中心点的x和y坐标
    center_x = int((p_left_x + p_right_x) / 2)
    center_y = int((p_top_y + p_bottom_y) / 2)
    # 确保边界在图像范围内
    height, width = depth_image.shape
    p_left_x = max(0, min(int(p_left_x), width - 1))
    p_right_x = max(0, min(int(p_right_x), width - 1))
    p_top_y = max(0, min(int(p_top_y), height - 1))
    p_bottom_y = max(0, min(int(p_bottom_y), height - 1))
    
    # 提取检测框内的深度区域
    roi_depth = depth_image[p_top_y:p_bottom_y, p_left_x:p_right_x]
    
    # 确保 roi_depth 不为空
    if roi_depth.size == 0:
        return (center_x, center_y), 2.0 

    # 过滤深度图，只保留检测框内0-max_distance米的点
    valid_depth_mask = (roi_depth > 0) & (roi_depth <= max_distance * 1000)
    
    # 提取有效的深度值
    valid_depths = roi_depth[valid_depth_mask]
    
    # 计算深度
    if len(valid_depths) > 0:
        # 使用 10% 分位数，只取前景（植物），忽略背景
        best_depth = np.percentile(valid_depths, 10) / 1000 
    else:
        # 兜底策略
        center_depth = depth_image[center_y, center_x]
        if 0 < center_depth <= max_distance * 1000:
            best_depth = center_depth / 1000
        else:
            best_depth = 2.0  # 默认值
    
    return (center_x, center_y), best_depth

def pixel_to_camera(K, pixel, Z):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y = pixel
    X_c = (x - cx) * Z / fx
    Y_c = (y - cy) * Z / fy
    return np.array([X_c, Y_c, Z])

def point_to_world(pixel, depth, translation, quaternion, intrinsic):
    cam_coord = pixel_to_camera(intrinsic.intrinsic_matrix, pixel, depth)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([cam_coord])
    world_pcd = ply_from_trajectory_forhuanong.transform_point_cloud(pcd, translation, quaternion)
    return world_pcd

def merge_fragmented_ids(id_set_list, id_pose_records, image_tree_list, merge_dist_threshold=1.2):
    """
    后处理：合并距离过近且没有时间冲突（不在同一帧出现）的ID
    """
    print("\n正在执行碎片ID合并...")
    
    # 1. 构建 ID -> 文件名集合 的映射
    id_files = {}
    for id_str, records in id_pose_records.items():
        id_files[id_str] = set([r[0] for r in records])

    changed = True
    while changed:
        changed = False
        current_ids = list(id_set_list.keys())
        
        for i in range(len(current_ids)):
            id_a = current_ids[i]
            if id_a not in id_set_list: continue
            center_a = np.asarray(id_set_list[id_a].id_center_3d.points)[0]
            
            for j in range(i + 1, len(current_ids)):
                id_b = current_ids[j]
                if id_b not in id_set_list: continue
                center_b = np.asarray(id_set_list[id_b].id_center_3d.points)[0]
                
                # 计算距离
                dist = np.linalg.norm(center_a - center_b)
                
                if dist < merge_dist_threshold:
                    # 检查冲突
                    common_files = id_files[id_a].intersection(id_files[id_b])
                    if len(common_files) == 0:
                        print(f"  -> 合并: ID {id_b} 合并入 {id_a} (距离: {dist:.3f}m)")
                        
                        # 合并数据
                        id_pose_records[id_a].extend(id_pose_records[id_b])
                        id_files[id_a].update(id_files[id_b])
                        
                        new_center = (center_a + center_b) / 2
                        id_set_list[id_a].id_center_3d.points = o3d.utility.Vector3dVector([new_center])
                        center_a = new_center 
                        
                        del id_set_list[id_b]
                        del id_pose_records[id_b]
                        
                        for img_tree in image_tree_list:
                            for idbox in img_tree.pr_tree:
                                if idbox.id == id_b:
                                    idbox.id = id_a
                        
                        changed = True
                        break 
            if changed: break
    print("合并完成。\n")

def save_plant_camera_coordinates(id_set_list, id_pose_records, output_dir):
    # (此函数保持不变，用于保存最终结果)
    coords_dir = os.path.join(output_dir, "plant_camera_coordinates")
    os.makedirs(coords_dir, exist_ok=True)
    
    plant_coords_file = os.path.join(coords_dir, "plant_centers_xyz.txt")
    with open(plant_coords_file, 'w', encoding='utf-8') as f:
        f.write("植物中心点坐标 (完整XYZ坐标)\n")
        f.write("=" * 80 + "\n\n")
        f.write("格式: ID, X(m), Y(m), Z(m)\n")
        f.write("-" * 60 + "\n")
        
        for plant_id, id_set in id_set_list.items():
            if hasattr(id_set, 'id_center_3d') and id_set.id_center_3d is not None:
                plant_points = np.asarray(id_set.id_center_3d.points)
                if len(plant_points) > 0:
                    x, y, z = plant_points[0]
                    f.write(f"ID {plant_id}: {x:.6f}, {y:.6f}, {z:.6f}\n")
                else:
                    f.write(f"ID {plant_id}: 无坐标数据\n")
            else:
                f.write(f"ID {plant_id}: 无id_center_3d属性\n")
    print(f"植物中心点坐标已保存到: {plant_coords_file}")
    
    camera_coords_file = os.path.join(coords_dir, "camera_poses_xyz.txt")
    with open(camera_coords_file, 'w', encoding='utf-8') as f:
        f.write("相机位姿坐标 (完整XYZ坐标 + 四元数朝向)\n")
        f.write("=" * 80 + "\n\n")
        f.write("格式: Plant_ID, Filename, X(m), Y(m), Z(m), qw, qx, qy, qz\n")
        f.write("-" * 60 + "\n")
        for plant_id, pose_records in id_pose_records.items():
            if pose_records:
                f.write(f"\n植物 ID: {plant_id}\n")
                f.write("-" * 40 + "\n")
                for filename, translation, quaternion in pose_records:
                    x, y, z = translation
                    qw, qx, qy, qz = quaternion
                    f.write(f"  {filename}: {x:.6f}, {y:.6f}, {z:.6f}, {qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}\n")
    print(f"相机位姿坐标已保存到: {camera_coords_file}")
    
    for plant_id, id_set in id_set_list.items():
        if plant_id in id_pose_records and id_pose_records[plant_id]:
            plant_coord_dir = os.path.join(coords_dir, f"plant_{plant_id}")
            os.makedirs(plant_coord_dir, exist_ok=True)
            plant_detail_file = os.path.join(plant_coord_dir, f"plant_{plant_id}_center.txt")
            with open(plant_detail_file, 'w', encoding='utf-8') as f:
                f.write(f"植物 {plant_id} 中心点坐标\n")
                f.write("=" * 60 + "\n\n")
                if hasattr(id_set, 'id_center_3d') and id_set.id_center_3d is not None:
                    plant_points = np.asarray(id_set.id_center_3d.points)
                    if len(plant_points) > 0:
                        x, y, z = plant_points[0]
                        f.write(f"X: {x:.6f} m\n")
                        f.write(f"Y: {y:.6f} m\n")
                        f.write(f"Z: {z:.6f} m\n")
                        camera_positions = []
                        for filename, translation, quaternion in id_pose_records[plant_id]:
                            camera_positions.append(translation)
                        if camera_positions:
                            cam_array = np.array(camera_positions)
                            avg_x = np.mean(cam_array[:, 0])
                            avg_y = np.mean(cam_array[:, 1])
                            avg_z = np.mean(cam_array[:, 2])
                            f.write(f"\n该植物对应的相机平均位置:\n")
                            f.write(f"X: {avg_x:.6f} m\n")
                            f.write(f"Y: {avg_y:.6f} m\n")
                            f.write(f"Z: {avg_z:.6f} m\n")
            camera_detail_file = os.path.join(plant_coord_dir, f"plant_{plant_id}_cameras.txt")
            with open(camera_detail_file, 'w', encoding='utf-8') as f:
                f.write(f"植物 {plant_id} 对应的相机坐标\n")
                f.write("=" * 60 + "\n\n")
                for i, (filename, translation, quaternion) in enumerate(id_pose_records[plant_id]):
                    x, y, z = translation
                    qw, qx, qy, qz = quaternion
                    f.write(f"相机 {i} ({filename}):\n")
                    f.write(f"  位置: ({x:.6f}, {y:.6f}, {z:.6f})\n")
                    f.write(f"  朝向: ({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})\n\n")
    create_summary_csv(id_set_list, id_pose_records, coords_dir)

def create_summary_csv(id_set_list, id_pose_records, coords_dir):
    plant_csv_file = os.path.join(coords_dir, "plant_centers_summary.csv")
    with open(plant_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Plant_ID', 'X(m)', 'Y(m)', 'Z(m)', 'Num_Cameras'])
        for plant_id, id_set in id_set_list.items():
            x, y, z = 0, 0, 0
            if hasattr(id_set, 'id_center_3d') and id_set.id_center_3d is not None:
                plant_points = np.asarray(id_set.id_center_3d.points)
                if len(plant_points) > 0:
                    x, y, z = plant_points[0]
            num_cameras = len(id_pose_records.get(plant_id, []))
            writer.writerow([plant_id, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", num_cameras])
    print(f"植物中心点汇总CSV已保存到: {plant_csv_file}")
    
    camera_csv_file = os.path.join(coords_dir, "camera_poses_summary.csv")
    with open(camera_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Plant_ID', 'Filename', 'X(m)', 'Y(m)', 'Z(m)', 'qw', 'qx', 'qy', 'qz'])
        for plant_id, pose_records in id_pose_records.items():
            for filename, translation, quaternion in pose_records:
                x, y, z = translation
                qw, qx, qy, qz = quaternion
                writer.writerow([
                    plant_id, filename, 
                    f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                    f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}"
                ])
    print(f"相机位姿汇总CSV已保存到: {camera_csv_file}")

# -----------------------------
# Main
# -----------------------------

def main(datapath):
    start_time = time.time() # [新增] 开始计时
    print(f"任务开始: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    model = YOLOWorld("weigths/best_forhuanong.pt")
    associations_file = os.path.join(datapath, 'associations.txt')
    associations = ply_from_trajectory_forhuanong.load_associations(associations_file)
    trajectory_file = os.path.join(datapath, 'CameraTrajectory.txt')
    trajectory = ply_from_trajectory_forhuanong.load_trajectory(trajectory_file)
    
    input_dir = os.path.join(datapath, "color")
    depth_dir = os.path.join(datapath, "depth")
    output_dir = os.path.join(datapath, "idout")
    os.makedirs(output_dir, exist_ok=True)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720,
        fx=813.19214658, fy=813.62230969,
        cx=651.42157643, cy=345.43288645
    )
    
    image_tree_list = []
    id_set_list = {}
    unassigned_images = []
    id_pose_records = {} 
    
    for filename in tqdm(sorted(os.listdir(input_dir))):
        if not filename.endswith(".png"):
            continue
        
        timestamp = associations.get(filename, None)
        if timestamp is None or timestamp not in trajectory:
            unassigned_images.append(filename)
            continue
            
        translation, quaternion = trajectory[timestamp]
        image_path = os.path.join(input_dir, filename)
        depth_path = os.path.join(depth_dir, filename.replace(".png", ".tif"))
        
        img = cv2.imread(image_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if img is None or depth_img is None:
            unassigned_images.append(filename)
            continue
        
        results = model(img, verbose=False, conf=0.79,iou=0.6)[0]
        boxes = np.asarray(results.boxes.xyxy.cpu())
        cls_tmp = np.asarray(results.boxes.cls.cpu())
        plant_boxes = boxes[cls_tmp == 0]
        
        if len(plant_boxes) == 0:
            unassigned_images.append(filename)
            continue
            
        draw_boxes_with_labels(img, plant_boxes, (255, 0, 0), "Plant")
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        cv2.imwrite(os.path.join(vis_dir, filename), img)
        
        image_tree = ImageTree(image_name=filename)
        
        # -----------------------------
        # 匹配 ID (Global Matching)
        # -----------------------------
        
        current_frame_plants = []
        for i, box in enumerate(plant_boxes):
            center_pixel, depth = get_point_center(box, depth_img)
            world_pcd = point_to_world(center_pixel, depth, translation, quaternion, intrinsic)
            current_frame_plants.append({
                "box": box,
                "center_pixel": center_pixel,
                "depth": depth,
                "world_pcd": world_pcd,
                "world_point": np.asarray(world_pcd.points)[0],
                "index": i
            })

        possible_matches = []
        MATCH_THRESHOLD = 1.5
        
        for p_idx, plant_info in enumerate(current_frame_plants):
            for existing_id, id_set in id_set_list.items():
                if hasattr(id_set, 'id_center_3d') and id_set.id_center_3d is not None:
                    existing_center = np.asarray(id_set.id_center_3d.points)[0]
                    dist = np.linalg.norm(plant_info["world_point"] - existing_center)
                    if dist < MATCH_THRESHOLD:
                        possible_matches.append((dist, p_idx, existing_id))

        possible_matches.sort(key=lambda x: x[0])

        assigned_box_indices = set()
        assigned_id_strs = set()

        # 1. 分配旧 ID
        for dist, p_idx, matched_id in possible_matches:
            if p_idx in assigned_box_indices: continue
            if matched_id in assigned_id_strs: continue
            
            assigned_box_indices.add(p_idx)
            assigned_id_strs.add(matched_id)
            
            plant_info = current_frame_plants[p_idx]
            id_set = id_set_list[matched_id]
            
            old_center = np.asarray(id_set.id_center_3d.points)[0]
            new_center = plant_info["world_point"]
            avg_center = old_center * 0.7 + new_center * 0.3 
            id_set.id_center_3d.points = o3d.utility.Vector3dVector([avg_center])
            
            # 【恢复日志输出】 
            with open(os.path.join(output_dir,'output.txt'), 'a') as f:
                 f.write(f"{matched_id} is {plant_info['center_pixel']} and {plant_info['depth']} and {np.asarray(plant_info['world_pcd'].points)} and {dist} and {filename}\n")
            
            id_pose_records[matched_id].append((filename, translation, quaternion))
            idboxtree = IDBoxTree(id=matched_id,
                                  id_center=plant_info["center_pixel"],
                                  goal_center=None, direction=None,
                                  idbox=plant_info["box"], goalbox=None)
            image_tree.add_id_box_tree(idboxtree)

        # 2. 分配新 ID
        for p_idx, plant_info in enumerate(current_frame_plants):
            if p_idx not in assigned_box_indices:
                new_id = f"{len(id_set_list)}"
                assigned_box_indices.add(p_idx)
                
                # 【恢复日志输出】
                with open(os.path.join(output_dir,'output.txt'), 'a') as f:
                    f.write(f"{new_id} is {plant_info['center_pixel']} and {plant_info['depth']} and {np.asarray(plant_info['world_pcd'].points)} and {filename}\n")
                
                id_set_list[new_id] = IDSET(
                    id=new_id, id_center_3d=plant_info["world_pcd"],
                    goal_center_3d=None, id_num=1, goal_num=0
                )
                id_pose_records[new_id] = []
                id_pose_records[new_id].append((filename, translation, quaternion))
                
                idboxtree = IDBoxTree(id=new_id,
                                      id_center=plant_info["center_pixel"],
                                      goal_center=None, direction=None,
                                      idbox=plant_info["box"], goalbox=None)
                image_tree.add_id_box_tree(idboxtree)

        image_tree_list.append(image_tree)

    for id_str, id_set in id_set_list.items():
        qr_dir = os.path.join(output_dir, id_str)
        color_dir = os.path.join(qr_dir, "color")
        depth_subdir = os.path.join(qr_dir, "depth")
        mask_dir = os.path.join(qr_dir, "mask")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_subdir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        id_traj_file = os.path.join(qr_dir, f"{id_str}_CameraTrajectory.txt")
        with open(id_traj_file, "w") as ftraj:
            for (fname, trans, quat) in id_pose_records[id_str]:
                ftraj.write(f"{fname} {trans[0]} {trans[1]} {trans[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")

        csv_filename = os.path.join(output_dir, f"{id_str}_image_namelist.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Image Name'])
            image_names = set()
            for (fname, trans, quat) in id_pose_records[id_str]:
                image_names.add(fname)
            for image_name in sorted(image_names):
                csv_writer.writerow([image_name])

        for image_tree in image_tree_list:
            filename = image_tree.image_name
            for idboxtree in image_tree.pr_tree:
                if idboxtree.id != id_str:
                    continue
                copyfile(os.path.join(input_dir, filename), os.path.join(color_dir, filename))
                depth_filename = filename.replace(".png", ".tif")
                copyfile(os.path.join(depth_dir, depth_filename), os.path.join(depth_subdir, depth_filename))
                
                binary_mask = np.zeros_like(depth_img, dtype=np.uint8)
                x1, y1, x2, y2 = map(int, idboxtree.idbox)
                cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)
                cv2.imwrite(os.path.join(mask_dir, filename), binary_mask * 255)

    unassigned_file = os.path.join(output_dir, "unassigned_images.txt")
    with open(unassigned_file, "w") as f:
        for img_name in unassigned_images:
            f.write(img_name + "\n")
            
    print("Processing completed.")
    
    # 后处理：合并碎片 ID
    merge_fragmented_ids(id_set_list, id_pose_records, image_tree_list, merge_dist_threshold=1.2)
    
    print("\n保存植物和相机的完整XYZ坐标...")
    try:
        save_plant_camera_coordinates(id_set_list, id_pose_records, output_dir)
        print("坐标保存完成")
    except Exception as e:
        print(f"坐标保存失败: {e}")

    # -----------------------------
    # [新增] 统计时间和GPU占用并保存
    # -----------------------------
    end_time = time.time()
    total_seconds = end_time - start_time
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    time_str = "{:d}h {:02d}m {:02d}s".format(int(h), int(m), int(s))
    
    stats_file = os.path.join(output_dir, "run_stats.txt")
    with open(stats_file, "w", encoding='utf-8') as f:
        f.write("运行统计报告\n")
        f.write("========================\n")
        f.write(f"总运行时间: {time_str} ({total_seconds:.2f} 秒)\n")
        
        # 获取GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # 获取当前占用的显存 (MB)
            curr_mem = torch.cuda.memory_allocated(0) / 1024 / 1024
            # 获取过程中的峰值显存 (MB)
            max_mem = torch.cuda.max_memory_allocated(0) / 1024 / 1024
            
            f.write(f"GPU 设备: {gpu_name}\n")
            f.write(f"当前显存占用: {curr_mem:.2f} MB\n")
            f.write(f"峰值显存占用: {max_mem:.2f} MB\n")
        else:
            f.write("GPU状态: 未检测到CUDA设备，使用CPU运行。\n")

    print(f"\n[Stats] 运行统计已保存至: {stats_file}")
    print(f"[Stats] 总耗时: {time_str}")
        
if __name__ == "__main__":

    #运行代码前请修改数据集路径
    main("dataset/huanong2")