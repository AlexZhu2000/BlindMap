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


def create_3d_box_corners(position, length, width, height, rotation):
    """
    Creates 8 corners of a 3D bounding box given its position, dimensions, and rotation.
    
    Args:
        position (numpy.ndarray): Center position of the box (x, y, z)
        length (float): Length of the box (along x-axis before rotation)
        width (float): Width of the box (along y-axis before rotation)
        height (float): Height of the box (along z-axis)
        rotation (float): Rotation angle in radians (around z-axis)
    
    Returns:
        numpy.ndarray: Array of shape (8, 3) containing the coordinates of the 8 corners
    """
    # Half dimensions
    l_2, w_2, h_2 = length / 2, width / 2, height / 2
    
    # Create corners assuming the box is aligned with the coordinate system
    corners = np.array([
        [l_2, w_2, -h_2],  # Front right top
        [l_2, -w_2, -h_2],  # Front left top
        [-l_2, -w_2, -h_2],  # Rear left top
        [-l_2, w_2, -h_2],  # Rear right top
        [l_2, w_2, h_2],   # Front right bottom
        [l_2, -w_2, h_2],   # Front left bottom
        [-l_2, -w_2, h_2],   # Rear left bottom
        [-l_2, w_2, h_2]    # Rear right bottom
    ])
    
    # Create rotation matrix around z-axis
    cos_rot, sin_rot = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([
        [cos_rot, -sin_rot, 0],
        [sin_rot, cos_rot, 0],
        [0, 0, 1]
    ])
    
    # Rotate corners
    rotated_corners = np.array([np.dot(rotation_matrix, corner) for corner in corners])
    
    # Translate corners to the box's position
    translated_corners = rotated_corners + position
    
    return translated_corners
def calculate_world2ego_occlusion_with_frustum(target_obj, other_objects, ego_position=(0, 0, 0)):
    """使用视锥体检测遮挡"""
    """
    使用视锥体检测遮挡，并提供更robust的实现
    
    Args:
        target_obj (dict): 目标物体信息
        other_objects (list): 场景中其他物体
        ego_position (tuple): 自车位置
        debug (bool): 是否开启调试模式
    
    Returns:
        int: 遮挡状态 (0: 无遮挡, 1: 部分遮挡, 2: 严重遮挡)
    """
    if not all(key in target_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
        print('no 3d_location')
        return 0
    
    # 获取目标物体位置和尺寸
    target_pos = np.array([
        float(target_obj['3d_location']['x']), 
        float(target_obj['3d_location']['y']),
        float(target_obj['3d_location']['z'])
    ])
    
    if not all(key in target_obj.get('3d_dimensions', {}) for key in ['l', 'w', 'h']):
        return 0
        
    target_length = float(target_obj['3d_dimensions']['l'])
    target_width = float(target_obj['3d_dimensions']['w'])
    target_height = float(target_obj['3d_dimensions']['h'])
    target_rotation = float(target_obj.get('rotation', 0))
    
    # 创建目标物体的边界框
    target_corners = create_3d_box_corners(target_pos, target_length, target_width, target_height, target_rotation)
    # 创建从自车到目标物体的视锥体
    frustum_vertices = create_view_frustum(ego_position, target_corners)
    
    # 检查其他物体是否与视锥体相交
    occluding_area = 0.0
    target_area = target_width * target_height  # 近似正视图面积
    
    for other_obj in other_objects:
        if other_obj == target_obj:
            continue
            
        if not all(key in other_obj.get('3d_location', {}) for key in ['x', 'y', 'z']):
            continue
        
        other_pos = np.array([
            float(other_obj['3d_location']['x']), 
            float(other_obj['3d_location']['y']),
            float(other_obj['3d_location']['z'])
        ])
        
        # 计算自车到物体的距离
        target_distance = np.linalg.norm(target_pos - ego_position)
        other_distance = np.linalg.norm(other_pos - ego_position)
        
        # 只考虑在自车和目标之间的物体
        if other_distance >= target_distance:
            # print('far than target_distance')
            continue
        
        # 获取物体的尺寸和旋转角度
        if not all(key in other_obj.get('3d_dimensions', {}) for key in ['l', 'w', 'h']):
            continue
            
        other_length = float(other_obj['3d_dimensions']['l'])
        other_width = float(other_obj['3d_dimensions']['w'])
        other_height = float(other_obj['3d_dimensions']['h'])
        other_rotation = float(other_obj.get('rotation', 0))
        # print('check if other_obj is in target_obj')
        # 创建物体的边界框
        other_corners = create_3d_box_corners(other_pos, other_length, other_width, other_height, other_rotation)
        
        # 检查物体是否与视锥体相交
        if frustum_box_intersection(frustum_vertices, other_corners):
            # 计算在视锥体中的投影面积
            projected_area = calculate_projected_area(other_obj, target_obj, ego_position)
            occluding_area += projected_area
    
    # 计算遮挡比例
    occlusion_ratio = min(1.0, occluding_area / target_area)
    
    # 根据遮挡比例确定遮挡状态
    if occlusion_ratio < 0.1:
        return 0  # 无明显遮挡
    elif occlusion_ratio < 0.5:
        return 1  # 部分遮挡
    else:
        return 2  # 严重遮挡

def create_view_frustum(eye_position, target_corners):
    """创建从视点到目标的视锥体"""
    # 视锥体由视点和目标边界框的8个角点确定
    eye = np.array(eye_position)
    frustum_vertices = [eye]
    
    for corner in target_corners:
        frustum_vertices.append(corner)
    
    return np.array(frustum_vertices)

def line_segment_triangle_intersection(p1, p2, t1, t2, t3):
    """
    Check if a line segment intersects a triangle.
    
    Args:
        p1, p2: Line segment endpoints
        t1, t2, t3: Triangle vertices
    
    Returns:
        bool: True if the line segment intersects the triangle
    """
    # Compute triangle normal
    edge1 = t2 - t1
    edge2 = t3 - t1
    normal = np.cross(edge1, edge2)
    normal_length = np.linalg.norm(normal)
    
    # Check for degenerate triangle
    if normal_length < 1e-6:
        return False
    
    normal = normal / normal_length
    
    # Compute line direction
    dir = p2 - p1
    dir_length = np.linalg.norm(dir)
    
    # Check for degenerate line segment
    if dir_length < 1e-6:
        return False
        
    dir = dir / dir_length
    
    # Check if line and triangle are parallel
    dot = np.dot(normal, dir)
    if abs(dot) < 1e-6:
        return False
    
    # Compute distance from p1 to triangle plane
    d = np.dot(normal, t1 - p1) / dot
    
    # Check if intersection point is beyond line segment
    if d < 0 or d > dir_length:
        return False
    
    # Compute intersection point
    intersection = p1 + d * dir
    
    # Check if intersection point is inside triangle
    vec0 = t1 - intersection
    vec1 = t2 - intersection
    vec2 = t3 - intersection
    
    # Cross products
    c0 = np.cross(vec0, vec1)
    c1 = np.cross(vec1, vec2)
    c2 = np.cross(vec2, vec0)
    
    # Check if intersection point is inside triangle
    if (np.dot(c0, normal) >= 0 and 
        np.dot(c1, normal) >= 0 and 
        np.dot(c2, normal) >= 0):
        return True
    
    return False
def frustum_box_intersection(frustum_vertices, box_corners):
    """
    Checks if a box intersects with a view frustum.
    
    Args:
        frustum_vertices: List of points defining the frustum (first point is eye, rest are box corners)
        box_corners: 8 corners of the box to check
    
    Returns:
        bool: True if the box intersects with the frustum
    """
    eye = frustum_vertices[0]
    target_corners = frustum_vertices[1:]
    
    # 1. Simple check: Is any corner of the box inside the frustum?
    for corner in box_corners:
        # Create rays from eye to target corners and check if the corner is inside
        inside = True
        for i in range(len(target_corners)):
            # Create a plane from eye and two adjacent target corners
            j = (i + 1) % len(target_corners)
            k = (i + 2) % len(target_corners)
            
            plane_normal = np.cross(target_corners[j] - eye, target_corners[k] - eye)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            
            # Check which side of the plane the corner is on
            side = np.dot(corner - eye, plane_normal)
            
            # If it's on the outside of any plane, it's outside the frustum
            if side < 0:
                inside = False
                break
        
        if inside:
            return True
    
    # 2. Check if any edge of the box intersects with any of the frustum's faces
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Create triangular faces of the frustum
    faces = []
    # Add triangular faces from eye to each pair of adjacent corners
    for i in range(len(target_corners)):
        j = (i + 1) % len(target_corners)
        faces.append([eye, target_corners[i], target_corners[j]])
    
    # Check if any edge intersects any face
    for edge_indices in box_edges:
        p1 = box_corners[edge_indices[0]]
        p2 = box_corners[edge_indices[1]]
        
        for face in faces:
            if line_segment_triangle_intersection(p1, p2, face[0], face[1], face[2]):
                return True
    
    # 3. Check if any edge of the frustum intersects with any face of the box
    # For simplicity, we'll skip this check as it's less common and more complex
    
    return False
def point_in_frustum(point, eye, faces):
    """检查点是否在视锥体内"""
    # 简化：检查点是否在所有视锥体面的正面
    for face in faces:
        if len(face) >= 3:
            v1 = face[1] - face[0]
            v2 = face[2] - face[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # 检查点是否在面的正面
            if np.dot(point - face[0], normal) < 0:
                return False
    
    return True

def line_triangle_intersection(line_start, line_end, triangle):
    """检查线段是否与三角形相交"""
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    
    # 计算三角形法向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    # 计算线段方向向量
    line_dir = line_end - line_start
    line_length = np.linalg.norm(line_dir)
    line_dir = line_dir / line_length
    
    # 检查线段是否与三角形所在平面平行
    dot_product = np.dot(line_dir, normal)
    if abs(dot_product) < 1e-6:
        return False
    
    # 计算线段与平面的交点
    d = np.dot(normal, v0 - line_start) / dot_product
    
    # 检查交点是否在线段上
    if d < 0 or d > line_length:
        return False
    
    # 计算交点坐标
    intersection = line_start + d * line_dir
    
    # 检查交点是否在三角形内
    edge0 = v1 - v0
    edge1 = v2 - v1
    edge2 = v0 - v2
    
    c0 = np.cross(intersection - v0, edge0)
    c1 = np.cross(intersection - v1, edge1)
    c2 = np.cross(intersection - v2, edge2)
    
    return (np.dot(c0, normal) >= 0 and 
            np.dot(c1, normal) >= 0 and 
            np.dot(c2, normal) >= 0)

def calculate_projected_area(occluder_obj, target_obj, ego_position):
    """计算遮挡物在目标方向上的投影面积"""
    # 简化：使用遮挡物的宽和高的投影
    occluder_width = float(occluder_obj['3d_dimensions']['w'])
    occluder_height = float(occluder_obj['3d_dimensions']['h'])
    
    occluder_pos = np.array([
        float(occluder_obj['3d_location']['x']), 
        float(occluder_obj['3d_location']['y']),
        float(occluder_obj['3d_location']['z'])
    ])
    
    target_pos = np.array([
        float(target_obj['3d_location']['x']), 
        float(target_obj['3d_location']['y']),
        float(target_obj['3d_location']['z'])
    ])
    
    ego_pos = np.array(ego_position)
    
    # 计算遮挡物到自车的距离
    occluder_distance = np.linalg.norm(occluder_pos - ego_pos)
    target_distance = np.linalg.norm(target_pos - ego_pos)
    
    # 根据距离比例调整投影面积
    # 物体越远，投影越小
    scale_factor = (target_distance / occluder_distance) ** 2
    
    # 计算方向向量，用于调整投影
    ego_to_target = target_pos - ego_pos
    ego_to_target = ego_to_target / np.linalg.norm(ego_to_target)
    
    ego_to_occluder = occluder_pos - ego_pos
    ego_to_occluder = ego_to_occluder / np.linalg.norm(ego_to_occluder)
    
    # 计算角度因子，如果物体不在视线上，投影面积减小
    angle_factor = np.dot(ego_to_target, ego_to_occluder)
    angle_factor = max(0.0, angle_factor) ** 2  # 二次方强调角度影响
    
    # 估算投影面积
    projected_area = occluder_width * occluder_height * scale_factor * angle_factor
    print('projected_area: ', projected_area)
    return projected_area
def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

import sys
import os
# 获取当前脚本所在目录的上级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 添加上级目录到 sys.path
sys.path.append(parent_dir)

from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils import box_utils
def check_and_complement_labels():
    # 定义路径
    root_path = '/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/'
    original_occluded_dir = "/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/DAIR-V2X-C_Complemented_Anno/zzh_with_occlusion/vehicle-side_label/lidar"
    luyifan_no_occluded_dir = os.path.join(root_path, "cooperative/label_world")
    output_dir = os.path.join(root_path, "cooperative/label_world_with_occlusion2")

    # 设置位置匹配的阈值（单位：米）
    POSITION_THRESHOLD = 0.1

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件名
    vehicle_side_files = os.listdir(original_occluded_dir)
    cooperative_label_files = os.listdir(luyifan_no_occluded_dir)

    # 遍历所有新标签文件
    # Process all new label files
    count = 0
    for filename in tqdm(cooperative_label_files):
        # count += 1
        # if count > 10:
        #     break
        print('filename:   ', filename)
        if not filename.endswith('.json'):
            print(f'not {filename} json, continue......')
            continue

        cooperative_label_path = os.path.join(luyifan_no_occluded_dir, filename)
        vehicle_side_path = os.path.join(original_occluded_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read new labels and backup labels
        new_cooperative_labels = read_json(cooperative_label_path)
        # print([obj['occluded_state'] for obj in new_cooperative_labels])
        # [None, None, None, None, None, None, None, None, None, '0', '1', '0', '2', '1', '1', '1', '0', '0']
        all_cooperative_objects_type = {id : obj['type'] for id, obj in enumerate(new_cooperative_labels)}
        # 按照vehicle_side_labels的长度和顺序将cooperative_labels的type赋值
        if filename in vehicle_side_files:
            vehicle_side_labels = read_json(vehicle_side_path)
            lidar_to_novatel_json = load_json(os.path.join(root_path, 'vehicle-side/calib/lidar_to_novatel/'+filename)) 
            novatel_to_world_json = load_json(os.path.join(root_path, 'vehicle-side/calib/novatel_to_world/'+filename))
            transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(
                                                lidar_to_novatel_json,
                                                novatel_to_world_json
                                            )
            ego_in_world = box_utils.zzh_project_single_point_by_matrix_torch(
                np.array([0, 0, 0]),
                transformation_matrix
            )
            all_vehicle_side_objects_type = {id : obj['type'] for id, obj in enumerate(vehicle_side_labels)}
            all_vehicle_side_objects_type_str = ''.join(all_vehicle_side_objects_type.values())
            all_cooperative_objects_type_str = ''.join(all_cooperative_objects_type.values())
            if all_vehicle_side_objects_type_str in all_cooperative_objects_type_str:
                for id, obj in enumerate(vehicle_side_labels):
                    new_cooperative_labels[id]['occluded_state'] = obj['occluded_state']
                # print('all_vehicle_side_objects_type_str in all_cooperative_objects_type_str')
            else:
            # Save problematic filenames to a text file
                with open("problematic_filenames.txt", "a") as f:
                    f.write(f"{filename}\n")
                print('all_vehicle_side_objects_type_str not in all_cooperative_objects_type_str')
            # print([obj['occluded_state'] for obj in new_cooperative_labels])
            # [2, 0, 0, 0, 1, 0, 0, 0, 2, '0', '1', '0', '2', '1', '1', '1', '0', '0']
            # print([obj['occluded_state'] for obj in vehicle_side_labels])
            # [2, 0, 0, 0, 1, 0, 0, 0, 2]
            # 将cooperative——labels中多余的obj按照光锥检测法检测其occulde_state，比较与Luyifan版本得到状态是否一致。
            cooperative_labels_backup_dict = new_cooperative_labels
            for id in range(len(vehicle_side_labels), len(new_cooperative_labels)):
                new_cooperative_labels[id]['occluded_state'] = '2'
                # obj = new_cooperative_labels[id]
                # other_objs_temp = cooperative_labels_backup_dict[:id]  + cooperative_labels_backup_dict[id+1:]
               
                # occlusion_state = calculate_world2ego_occlusion_with_frustum(obj, other_objs_temp, ego_in_world)
                # print(f'original occlusion state:{new_cooperative_labels[id]["occluded_state"]}, \t, \
                #         zzh caculated occlusion state:{occlusion_state}')
                # obj['occluded_state'] = occlusion_state

        else:
            # No backup file, calculate occlusion for all objects
            print('No backup file, calculate occlusion for all objects')
            for obj in new_cooperative_labels:
                # Use 3D method since 2D info might not be available
                obj['occluded_state'] = calculate_world2ego_occlusion_with_frustum(obj, new_cooperative_labels, ego_in_world)

        # Write processed labels to output directory
        write_json(output_path, new_cooperative_labels)

def test_occlusion_detection():
    # Create a simple scene
    ego_position = np.array([0, 0, 0])
    
    # Create a target object
    target_obj = {
        '3d_location': {'x': 10.0, 'y': 0.0, 'z': 0.0},
        '3d_dimensions': {'l': 2.0, 'w': 2.0, 'h': 2.0},
        'rotation': 0.0
    }
    
    # Create an occluding object directly in the line of sight
    occluder_obj = {
        '3d_location': {'x': 5.0, 'y': 0.0, 'z': 0.0},
        '3d_dimensions': {'l': 1.0, 'w': 1.0, 'h': 1.0},
        'rotation': 0.0
    }
    
    # Create a non-occluding object to the side
    side_obj = {
        '3d_location': {'x': 5.0, 'y': 5.0, 'z': 0.0},
        '3d_dimensions': {'l': 1.0, 'w': 1.0, 'h': 1.0},
        'rotation': 0.0
    }
    
    # Get occlusion state for the target with the occluder
    scene1 = [target_obj, occluder_obj]
    occlusion1 = calculate_occlusion_with_frustum(target_obj, scene1, ego_position)
    print(f"Occlusion with occluder in line of sight: {occlusion1}")
    
    # Get occlusion state for the target with the side object
    scene2 = [target_obj, side_obj]
    occlusion2 = calculate_occlusion_with_frustum(target_obj, scene2, ego_position)
    print(f"Occlusion with object to the side: {occlusion2}")
if __name__ == "__main__":
    check_and_complement_labels()