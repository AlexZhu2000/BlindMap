import numpy as np
from scipy.spatial import KDTree
def get_box_corners(vehicle):
    """将车辆参数转换为8个角点坐标
    
    Args:
        vehicle (dict): 包含'location'、'center'、'extent'、'angle'的车辆信息
        
    Returns:
        np.ndarray: 形状为(8, 3)的角点坐标数组
    """
    # 获取基本参数
    loc = vehicle['location']                      # [x, y, z]
    center = vehicle.get('center', [0, 0, 0])      # 可选的偏移量
    extent = vehicle['extent']                     # [length/2, width/2, height/2]
    angle = vehicle['angle']                       # [roll, pitch, yaw]
    corners = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])
    # # 在局部坐标系中创建盒子角点
    # l, w, h = extent                              # 半长度
    # corners = np.array([
    #     [ l,  w,  h],  # 前-右-上
    #     [ l,  w, -h],  # 前-右-下
    #     [ l, -w,  h],  # 前-左-上 
    #     [ l, -w, -h],  # 前-左-下
    #     [-l,  w,  h],  # 后-右-上
    #     [-l,  w, -h],  # 后-右-下
    #     [-l, -w,  h],  # 后-左-上
    #     [-l, -w, -h]   # 后-左-下
    # ])
    
    # 绕z轴旋转盒子(偏航角)
    yaw = angle[2]                                # 只使用偏航角
    rotation_z = np.array([
        [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
        [np.sin(np.radians(yaw)),  np.cos(np.radians(yaw)), 0],
        [0,                        0,                        1]
    ])
    corners = corners @ rotation_z.T
    
    # 转换到世界坐标系
    corners = corners + np.array([
        loc[0] + center[0],                       # 全局x
        loc[1] + center[1],                       # 全局y
        loc[2] + center[2]                        # 全局z
    ])
    
    return corners
def fast_occlusion_detection(target_vehicle, other_vehicles, sensor_position, num_rays=20):
    """
    基于光线投射的快速遮挡检测方法
    
    参数：
        target_vehicle (dict): 目标车辆信息
        other_vehicles (list): 其他车辆信息列表
        sensor_position (np.ndarray): 传感器位置 [x, y, z]
        num_rays (int): 用于检测的光线数量
        
    返回：
        float: 遮挡比例 [0-1]
        int: 遮挡状态 (0-无遮挡, 1-部分遮挡, 2-严重遮挡)
    """
    # 获取目标车辆的8个角点和中心点
    target_corners = get_box_corners(target_vehicle)
    target_center = np.array(target_vehicle['location'])
    
    # 定义车辆6个面的中心点
    face_centers = []
    
    # 前后面
    face_centers.append(target_center + np.array([target_vehicle['extent'][0], 0, 0]))
    face_centers.append(target_center - np.array([target_vehicle['extent'][0], 0, 0]))
    
    # 左右面
    face_centers.append(target_center + np.array([0, target_vehicle['extent'][1], 0]))
    face_centers.append(target_center - np.array([0, target_vehicle['extent'][1], 0]))
    
    # 上下面
    face_centers.append(target_center + np.array([0, 0, target_vehicle['extent'][2]]))
    face_centers.append(target_center - np.array([0, 0, target_vehicle['extent'][2]]))
    
    # 采样点: 包括角点、面中心点和均匀分布的额外点
    sample_points = []
    
    # 添加角点
    for corner in target_corners:
        sample_points.append(corner)
    
    # 添加面中心点
    for face_center in face_centers:
        sample_points.append(face_center)
    
    # 随机添加更多采样点直到达到指定数量
    while len(sample_points) < num_rays and len(sample_points) < 50:
        # 随机选择一个面
        face_idx = np.random.randint(0, 6)
        face_center = face_centers[face_idx]
        
        # 在面附近随机采样
        if face_idx < 2:  # 前后面 (yz平面)
            random_point = face_center + np.array([0, 
                                                  np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][1],
                                                  np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][2]])
        elif face_idx < 4:  # 左右面 (xz平面)
            random_point = face_center + np.array([np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][0],
                                                  0,
                                                  np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][2]])
        else:  # 上下面 (xy平面)
            random_point = face_center + np.array([np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][0],
                                                  np.random.uniform(-0.8, 0.8) * target_vehicle['extent'][1],
                                                  0])
        
        sample_points.append(random_point)
    
    # 对其他车辆进行预处理
    other_boxes = []
    for vehicle in other_vehicles:
        other_boxes.append(get_box_corners(vehicle))
    
    # 发射光线并检查遮挡
    occluded_rays = 0
    valid_rays = 0
    
    for point in sample_points:
        # 计算从传感器到采样点的方向
        direction = point - sensor_position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # 避免零距离
            continue
            
        direction = direction / distance
        
        # 检查此光线是否被其他车辆遮挡
        is_occluded = False
        for other_corners in other_boxes:
            # 使用简化的光线-盒子相交测试
            hit_distance = ray_box_intersection(sensor_position, direction, other_corners)
            
            # 如果光线与其他车辆相交，且相交点距离小于目标点距离
            if 0 < hit_distance < distance - 0.1:  # 0.1米的容差
                is_occluded = True
                break
        
        if is_occluded:
            occluded_rays += 1
        
        valid_rays += 1
    
    # 计算遮挡比例
    if valid_rays == 0:
        return 1.0, 2  # 默认为完全遮挡
        
    occlusion_ratio = occluded_rays / valid_rays
    
    # 判断遮挡状态
    if occlusion_ratio < 0.1:
        return occlusion_ratio, 0  # 无遮挡
    elif occlusion_ratio < 0.5:
        return occlusion_ratio, 1  # 部分遮挡
    else:
        return occlusion_ratio, 2  # 严重遮挡

def ray_box_intersection(origin, direction, box_corners):
    """
    优化的光线-盒子相交测试
    
    返回:
        float: 相交距离，如果不相交则返回 -1
    """
    # 1. 计算车辆的6个面
    faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 侧面1
        [2, 3, 7, 6],  # 侧面2
        [0, 3, 7, 4],  # 侧面3
        [1, 2, 6, 5]   # 侧面4
    ]
    
    min_distance = float('inf')
    has_intersection = False
    
    # 2. 对每个面进行光线相交测试
    for face in faces:
        # 分割成两个三角形
        triangles = [
            [box_corners[face[0]], box_corners[face[1]], box_corners[face[2]]],
            [box_corners[face[0]], box_corners[face[2]], box_corners[face[3]]]
        ]
        
        for triangle in triangles:
            # 计算Möller–Trumbore相交算法
            v0, v1, v2 = triangle
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(direction, edge2)
            a = np.dot(edge1, h)
            
            # 如果射线与三角形平行
            if abs(a) < 1e-6:
                continue
                
            f = 1.0 / a
            s = origin - v0
            u = f * np.dot(s, h)
            
            # 不在三角形内
            if u < 0.0 or u > 1.0:
                continue
                
            q = np.cross(s, edge1)
            v = f * np.dot(direction, q)
            
            # 不在三角形内
            if v < 0.0 or u + v > 1.0:
                continue
                
            # 计算交点距离
            t = f * np.dot(edge2, q)
            
            # 交点在光线方向上且距离更近
            if t > 1e-6 and t < min_distance:
                has_intersection = True
                min_distance = t
    
    if has_intersection:
        return min_distance
    else:
        return -1.0