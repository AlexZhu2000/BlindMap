import json
import os
import numpy as np
from tqdm import tqdm

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_distance(pos1, pos2):
    """计算两个3D位置之间的欧氏距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + 
                  (pos1[1] - pos2[1])**2 + 
                  (pos1[2] - pos2[2])**2)



from shapely.geometry import Point, LineString, Polygon
from shapely import intersects
def calculate_2d_box_overlap(box1, box2):
    """计算两个2D框的重叠率"""
    # 获取两个框的坐标
    x1_min, y1_min = box1['xmin'], box1['ymin']
    x1_max, y1_max = box1['xmax'], box1['ymax']
    x2_min, y2_min = box2['xmin'], box2['ymin']
    x2_max, y2_max = box2['xmax'], box2['ymax']
    
    # 计算重叠区域
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算重叠面积和框1的面积
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # 返回重叠率
    return overlap_area / box1_area
def calculate_3d_occlusion(target_obj, other_objects, ego_position=(0, 0, 0)):
    """Calculate occlusion based purely on 3D information"""
    if not all(key in target_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
        return 0
    
    target_pos = (float(target_obj['3d_location']['x']), 
                 float(target_obj['3d_location']['y']),
                 float(target_obj['3d_location']['z']))
    target_distance = calculate_distance(target_pos, ego_position)
    
    # Create a line from ego to target
    ego_to_target_vector = np.array([
        target_pos[0] - ego_position[0],
        target_pos[1] - ego_position[1],
        target_pos[2] - ego_position[2]
    ])
    ego_to_target_unit = ego_to_target_vector / np.linalg.norm(ego_to_target_vector)
    
    # Check for occlusions
    occlusion_count = 0
    for other_obj in other_objects:
        if other_obj == target_obj:
            continue
            
        if not all(key in other_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
            continue
        
        other_pos = (float(other_obj['3d_location']['x']), 
                    float(other_obj['3d_location']['y']),
                    float(other_obj['3d_location']['z']))
        other_distance = calculate_distance(other_pos, ego_position)
        
        # Only consider objects between ego and target
        if other_distance >= target_distance:
            continue
        
        # Check if other object is in the line of sight
        # Project other_pos onto the ego_to_target line
        other_to_ego_vector = np.array([
            ego_position[0] - other_pos[0],
            ego_position[1] - other_pos[1],
            ego_position[2] - other_pos[2]
        ])
        
        # Calculate the projected distance
        projection = np.dot(other_to_ego_vector, -ego_to_target_unit)
        
        # Calculate perpendicular distance from other object to line of sight
        perpendicular_distance = np.linalg.norm(
            other_to_ego_vector + projection * ego_to_target_unit
        )
        
        # Get the dimensions of the other object
        if '3d_dimensions' in other_obj:
            other_width = float(other_obj['3d_dimensions']['w'])
            other_height = float(other_obj['3d_dimensions']['h'])
            other_length = float(other_obj['3d_dimensions']['l'])
            
            # Simple approximation: if perpendicular distance is less than half diagonal of bounding box
            other_size = np.sqrt(other_width**2 + other_height**2 + other_length**2) / 2
            
            if perpendicular_distance < other_size:
                occlusion_count += 1
    
    # Determine occlusion state based on count
    if occlusion_count == 0:
        return 0  # No occlusion
    elif occlusion_count == 1:
        return 1  # Partial occlusion
    else:
        return 2  # Severe occlusion
def calculate_occlusion_new(target_obj, other_objects, ego_position=(0, 0, 0)):
    """基于2D框重叠和距离计算遮挡状态"""
    if not all(key in target_obj for key in ['2d_box', '3d_location']):
        return 0
    
    target_box = target_obj['2d_box']
    target_distance = calculate_distance(
        (float(target_obj['3d_location']['x']),
         float(target_obj['3d_location']['y']),
         float(target_obj['3d_location']['z'])),
        ego_position
    )
    
    max_overlap = 0.0
    for other_obj in other_objects:
        if other_obj == target_obj or '2d_box' not in other_obj:
            continue
            
        other_distance = calculate_distance(
            (float(other_obj['3d_location']['x']),
             float(other_obj['3d_location']['y']),
             float(other_obj['3d_location']['z'])),
            ego_position
        )
        
        # 只考虑比目标物体更近的物体
        if other_distance >= target_distance:
            continue
            
        overlap_ratio = calculate_2d_box_overlap(target_box, other_obj['2d_box'])
        max_overlap = max(max_overlap, overlap_ratio)
    
    # 根据重叠率确定遮挡状态
    if max_overlap < 0.1:  # 几乎没有重叠
        return 0
    elif max_overlap < 0.5:  # 部分重叠
        return 1
    else:  # 严重重叠
        return 2
def get_bbox_polygon(obj):
    """获取目标的2D边界框多边形"""
    center_x = float(obj['3d_location']['x'])
    center_y = float(obj['3d_location']['y'])
    length = float(obj['3d_dimensions']['l'])
    width = float(obj['3d_dimensions']['w'])
    rotation = float(obj['rotation'])
    
    # 计算边界框的四个角点
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    corners = []
    for dx, dy in [(-length/2, -width/2), (length/2, -width/2), 
                   (length/2, width/2), (-length/2, width/2)]:
        x = center_x + dx * cos_rot - dy * sin_rot
        y = center_y + dx * sin_rot + dy * cos_rot
        corners.append((x, y))
    
    return Polygon(corners)

    """计算目标物体的遮挡状态"""
    if not all(key in target_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
        return 0
    
    target_pos = Point(float(target_obj['3d_location']['x']), 
                      float(target_obj['3d_location']['y']))
    ego_point = Point(ego_position[0], ego_position[1])
    
    # 创建从ego到目标的视线
    sight_line = LineString([ego_point, target_pos])
    target_distance = ego_point.distance(target_pos)
    
    # 获取目标物体的边界框
    target_bbox = get_bbox_polygon(target_obj)
    
    # 检查其他物体是否遮挡视线
    occlusion_count = 0
    for other_obj in other_objects:
        if other_obj == target_obj:
            continue
            
        if not all(key in other_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
            continue
            
        other_bbox = get_bbox_polygon(other_obj)
        other_distance = ego_point.distance(Point(float(other_obj['3d_location']['x']),
                                                float(other_obj['3d_location']['y'])))
        
        # 只考虑在ego和目标之间的物体
        if other_distance >= target_distance:
            continue
            
        # 检查是否与视线相交
        if intersects(sight_line, other_bbox):
            occlusion_count += 1
    
    # 根据遮挡计数确定遮挡状态
    if occlusion_count == 0:
        return 0  # 无遮挡
    elif occlusion_count == 1:
        return 1  # 部分遮挡 (0-50%)
    else:
        return 2  # 严重遮挡 (50-100%)
def check_and_complement_labels():
    # 定义路径
    original_occluded_dir = "/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/label/lidar_backup"
    luyifan_no_occluded_dir = "/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/DAIR-V2X-C_Complemented_Anno/new_labels/vehicle-side_label/lidar"
    output_dir = "/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/DAIR-V2X-C_Complemented_Anno/zzh_with_occlusion/vehicle-side_label/lidar"

    # 设置位置匹配的阈值（单位：米）
    POSITION_THRESHOLD = 0.5

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件名
    backup_files = os.listdir(original_occluded_dir)
    new_label_files = os.listdir(luyifan_no_occluded_dir)

    # 遍历所有新标签文件
    # Process all new label files
    for filename in tqdm(new_label_files):
        print('filename:   ', filename)
        if not filename.endswith('.json'):
            print(f'not {filename} json, continue......')
            continue

        new_label_path = os.path.join(luyifan_no_occluded_dir, filename)
        backup_path = os.path.join(original_occluded_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read new labels and backup labels
        new_labels = read_json(new_label_path)
        
        # Check if backup file exists
        if filename in backup_files:
            backup_labels = read_json(backup_path)
            
            # Create a dictionary of backup labels for fast lookup
            backup_dict = {}
            for obj in backup_labels:
                if all(key in obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
                    pos = (float(obj['3d_location']['x']),
                          float(obj['3d_location']['y']),
                          float(obj['3d_location']['z']))
                    backup_dict[pos] = obj

            # Update labels
            for obj in new_labels:
                if all(key in obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
                    current_pos = (float(obj['3d_location']['x']),
                                 float(obj['3d_location']['y']),
                                 float(obj['3d_location']['z']))
                    
                    # Find matching backup object
                    min_dist = float('inf')
                    matching_backup = None
                    
                    for backup_pos, backup_obj in backup_dict.items():
                        dist = calculate_distance(current_pos, backup_pos)
                        if dist < min_dist and dist < POSITION_THRESHOLD:
                            min_dist = dist
                            matching_backup = backup_obj
                    
                    if matching_backup is not None:
                        print(obj)
                        # Check if types match
                        if obj['type'] != matching_backup['type']:
                            print('type does not match')
                            # Calculate occlusion using 3D method
                            obj['occluded_state'] = calculate_3d_occlusion(obj, new_labels)
                        else:
                            # Use existing occlusion state
                            obj['occluded_state'] = matching_backup['occluded_state']
                            # Copy 2d_box if available
                            if '2d_box' in matching_backup:
                                print('2d_box in matchjson')
                                obj['2d_box'] = matching_backup['2d_box']
                            else:
                                print('2d_box not in matchjson')
                    else:
                        # No matching backup, calculate new occlusion
                        # First try the 2D method if possible
                        if '2d_box' in obj and all('2d_box' in other_obj for other_obj in new_labels):
                            obj['occluded_state'] = calculate_occlusion_new(obj, new_labels)
                        else:
                            # Fall back to 3D method
                            obj['occluded_state'] = calculate_3d_occlusion(obj, new_labels)
                else:
                    # No 3D location, set default
                    obj['occluded_state'] = 0
        else:
            # No backup file, calculate occlusion for all objects
            for obj in new_labels:
                # Use 3D method since 2D info might not be available
                obj['occluded_state'] = calculate_3d_occlusion(obj, new_labels)

        # Write processed labels to output directory
        write_json(output_path, new_labels)
if __name__ == "__main__":
    check_and_complement_labels()