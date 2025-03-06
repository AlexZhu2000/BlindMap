import numpy as np
import json
import os
from tqdm import tqdm

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_3d_bbox_corners(dimensions, location, rotation):
    """
    获取3D边界框的8个角点坐标
    Args:
        dimensions: dict with 'l','w','h' keys
        location: dict with 'x','y','z' keys
        rotation: float, rotation around Z axis
    Returns:
        corners: (8,3) array
    """
    l, w, h = float(dimensions['l']), float(dimensions['w']), float(dimensions['h'])
    x, y, z = float(location['x']), float(location['y']), float(location['z'])
    
    # 创建3D边界框的8个角点（以中心为原点）
    corners = np.array([
        [l/2, w/2, h/2],  # 前右上
        [l/2, w/2, -h/2], # 前右下
        [l/2, -w/2, h/2], # 前左上
        [l/2, -w/2, -h/2],# 前左下
        [-l/2, w/2, h/2], # 后右上
        [-l/2, w/2, -h/2],# 后右下
        [-l/2, -w/2, h/2],# 后左上
        [-l/2, -w/2, -h/2]# 后左下
    ])
    
    # 绕Z轴旋转
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]
    ])
    
    # 旋转和平移
    corners = corners @ rot_matrix.T + np.array([x, y, z])
    
    return corners

def project_3d_to_2d(points_3d, calib_info):
    """
    将3D点投影到2D图像平面
    Args:
        points_3d: (N,3) array
        calib_info: 包含相机内外参的字典
    Returns:
        points_2d: (N,2) array
    """
    # 构建相机内参矩阵
    intrinsic = np.array(calib_info['intrinsic']).reshape(3,3)
    
    # 构建外参矩阵
    extrinsic = np.array(calib_info['extrinsic']).reshape(4,4)
    
    # 转换为齐次坐标
    points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # 坐标变换: 世界坐标 -> 相机坐标
    points_cam = points_h @ extrinsic.T
    
    # 投影到图像平面
    points_2d_h = points_cam[:, :3] @ intrinsic.T
    
    # 归一化
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    
    return points_2d

def get_2d_bbox(corners_2d):
    """计算2D边界框"""
    xmin = np.min(corners_2d[:, 0])
    ymin = np.min(corners_2d[:, 1])
    xmax = np.max(corners_2d[:, 0])
    ymax = np.max(corners_2d[:, 1])
    
    return {
        'xmin': float(xmin),
        'ymin': float(ymin),
        'xmax': float(xmax),
        'ymax': float(ymax)
    }

def process_labels(label_path, calib_path, img_shape):
    """
    处理标签文件，添加2D边界框
    Args:
        label_path: 标签文件路径
        calib_path: 标定文件路径
        img_shape: (H,W) 图像尺寸
    """
    # 读取标签和标定信息
    labels = read_json(label_path)
    calib_info = read_json(calib_path)
    
    for obj in labels:
        if not all(key in obj for key in ['3d_dimensions', '3d_location', 'rotation']):
            continue
            
        # 获取3D边界框角点
        corners_3d = get_3d_bbox_corners(
            obj['3d_dimensions'],
            obj['3d_location'],
            float(obj['rotation'])
        )
        
        # 投影到2D
        corners_2d = project_3d_to_2d(corners_3d, calib_info)
        
        # 检查是否在图像范围内
        if np.any((corners_2d < 0) | (corners_2d[:, 0] >= img_shape[1]) | 
                 (corners_2d[:, 1] >= img_shape[0])):
            continue
            
        # 计算2D边界框
        obj['2d_box'] = get_2d_bbox(corners_2d)
    
    return labels

def main():
    # 设置路径
    base_dir = "/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c"
    label_dir = f"{base_dir}/cooperative-vehicle-infrastructure/vehicle-side/label/lidar"
    calib_dir = f"{base_dir}/cooperative-vehicle-infrastructure/vehicle-side/calib"
    output_dir = f"{base_dir}/cooperative-vehicle-infrastructure/vehicle-side/label/lidar_with_2d"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像尺寸 (根据实际情况修改)
    img_shape = (1080, 1920)  # (H, W)
    
    # 处理所有标签文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    for filename in tqdm(label_files):
        label_path = os.path.join(label_dir, filename)
        calib_path = os.path.join(calib_dir, filename)
        
        if not os.path.exists(calib_path):
            print(f"Warning: Calibration file not found for {filename}")
            continue
            
        # 处理标签
        updated_labels = process_labels(label_path, calib_path, img_shape)
        
        # 保存结果
        output_path = os.path.join(output_dir, filename)
        write_json(output_path, updated_labels)

if __name__ == "__main__":
    main()