# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:29:53
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:29:53

import numpy as np
import os
import yaml
from scipy.spatial import ConvexHull
from tqdm import tqdm
import sys
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(project_root)
from occlusion_detection import get_box_corners,fast_occlusion_detection

def ray_intersects_box(origin, direction, box_corners):
    """ 
    Args:
        origin (np.ndarray): 射线起点
        direction (np.ndarray): 射线方向（单位向量）
        box_corners (np.ndarray): 盒子的角点坐标
        
    Returns:
        bool: 如果射线与盒子相交则返回True
    """
    try:
        hull = ConvexHull(box_corners)
        for simplex in hull.simplices:
            triangle = box_corners[simplex]
            # 简单的三角形相交测试
            v0 = triangle[1] - triangle[0]
            v1 = triangle[2] - triangle[0]
            normal = np.cross(v0, v1)
            
            # 跳过退化的三角形
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
                
            normal = normal / norm
            
            d = -np.dot(normal, triangle[0])
            denom = np.dot(normal, direction)
            
            # 避免除以零
            if abs(denom) < 1e-6:
                continue
                
            t = -(np.dot(normal, origin) + d) / denom
            
            if t > 0:  # 相交点在射线起点前方
                # 计算相交点
                intersection = origin + t * direction
                
                # 检查相交点是否在三角形内
                edge0 = triangle[1] - triangle[0]
                edge1 = triangle[2] - triangle[0]
                edge2 = triangle[0] - triangle[2]
                
                c0 = intersection - triangle[0]
                c1 = intersection - triangle[1]
                c2 = intersection - triangle[2]
                
                if (np.dot(np.cross(edge0, c0), normal) >= 0 and
                    np.dot(np.cross(edge1, c1), normal) >= 0 and
                    np.dot(np.cross(edge2, c2), normal) >= 0):
                    return True
    except Exception as e:
        # 记录异常信息但继续执行
        print(f"射线相交检测出错: {e}")
    return False

def calculate_vehicle_occlusion(target_vehicle, other_vehicles, observer_pos):
    """计算目标车辆的遮挡状态
    
    Args:
        target_vehicle (dict): 目标车辆信息
        other_vehicles (list): 其他车辆信息列表
        observer_pos (np.ndarray): 观察者位置
        
    Returns:
        int: 0-无遮挡, 1-部分遮挡, 2-严重遮挡
        float: 可见面积比例
    """
    # 获取目标车辆角点
    target_corners = get_box_corners(target_vehicle)
    
    # 定义车辆表面索引
    surfaces = [
        [0, 1, 2, 3],  # 前
        [4, 5, 6, 7],  # 后
        [0, 1, 4, 5],  # 右
        [2, 3, 6, 7],  # 左
        [0, 2, 4, 6],  # 上
        [1, 3, 5, 7]   # 下
    ]
    
    # 计算总面积和可见面积
    total_area = 0
    visible_area = 0
    
    # 预先计算所有其他车辆的角点，避免重复计算
    other_corners_list = [get_box_corners(other) for other in other_vehicles]
    
    for surface in surfaces:
        surface_corners = target_corners[surface]
        try:
            # 计算面的法向量来确定它的朝向
            v1 = surface_corners[1] - surface_corners[0]
            v2 = surface_corners[2] - surface_corners[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # 基于面的朝向选择合适的投影平面
            abs_normal = np.abs(normal)
            max_idx = np.argmax(abs_normal)
            
            if max_idx == 0:  # 面主要朝向x轴
                proj_corners = surface_corners[:, 1:]  # 投影到y-z平面
            elif max_idx == 1:  # 面主要朝向y轴
                proj_corners = surface_corners[:, [0, 2]]  # 投影到x-z平面
            else:  # 面主要朝向z轴
                proj_corners = surface_corners[:, :2]  # 投影到x-y平面
            
            # 计算投影面积
            try:
                hull = ConvexHull(proj_corners)
                # 根据法向量校正面积
                correction = 1.0 / abs(normal[max_idx]) if abs(normal[max_idx]) > 1e-6 else 1.0
                area = hull.area * correction
            except Exception as qhull_error:
                # 备用方法：计算对角线叉积的方法来估算面积
                diag1 = surface_corners[2] - surface_corners[0]
                diag2 = surface_corners[3] - surface_corners[1]
                area = np.linalg.norm(np.cross(diag1, diag2)) / 2
            
            total_area += area
            
            # 检查表面中心的可见性
            center = surface_corners.mean(axis=0)
            direction = center - observer_pos
            direction_norm = np.linalg.norm(direction)
            
            # 检查距离是否有效
            if direction_norm < 1e-6:
                continue
                
            direction = direction / direction_norm
            
            # 检查表面是否被其他车辆遮挡
            occluded = False
            for other_corners in other_corners_list:
                if ray_intersects_box(observer_pos, direction, other_corners):
                    occluded = True
                    break
            
            if not occluded:
                visible_area += area
                
        except Exception as e:
            print(f"计算表面可见性出错: {e}")
            continue
    
    # 计算可见比例
    visible_ratio = 0
    if total_area > 0:
        visible_ratio = visible_area / total_area
    
    # 分类遮挡状态
    if visible_ratio > 0.7:    # 可见超过70%
        return 0, visible_ratio
    elif visible_ratio > 0.3:  # 可见30%-70%
        return 1, visible_ratio
    else:                      # 可见少于30%
        return 2, visible_ratio
import re
def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param

def process_scene(agent_id_path):
    """处理一个场景的数据
    
    Args:
        agent_id_path (str): 代理ID的路径
    """
    # for timestamp in sorted(os.listdir(agent_id_path)):
    #     if "occluded_state" in timestamp:
    #         os.remove(os.path.join(agent_id_path, timestamp))
    #         print(f"删除文件 {timestamp} 成功")
    #         continue
    for timestamp in sorted(os.listdir(agent_id_path)):
        if not timestamp.endswith('.yaml'):
            continue
        if timestamp.endswith('occluded_state.yaml'):
            continue
        occluded_state_path = os.path.join(agent_id_path, timestamp.replace('.yaml', '_occluded_state.yaml'))
        if os.path.exists(occluded_state_path):
            continue  # 如果存在该文件，则跳过当前时间戳文件
        
        
        
        yaml_path = os.path.join(agent_id_path, timestamp)
        print(f'正在处理文件: {yaml_path}')
        data=load_yaml(yaml_path)

        # print(data)
        # 跳过RSU（路边单元）
        if data.get('RSU', False):
            continue
        
        # 获取雷达位置
        lidar_pose = data.get('lidar_pose')
        if lidar_pose is None:
            continue
            
        # 获取所有车辆
        vehicles = data.get('vehicles', [])
        if not vehicles:
            continue
            
        lidar_pos = np.array(lidar_pose[:3])
        
        # 计算每辆车的遮挡状态
        for vehicle_id, vehicle in vehicles.items():
            # 跳过缺少必要信息的车辆
            if not all(k in vehicle for k in ['location', 'angle', 'extent']):
                vehicle['occluded_state'] = 0  # 默认为无遮挡
                continue
                
            # 获取当前车辆以外的所有车辆
            other_vehicles_list = [veh for vid, veh in vehicles.items() if vid != vehicle_id]
            
            # 计算遮挡状态
            # occlusion_state, visible_ratio = calculate_vehicle_occlusion(
            #     vehicle, other_vehicles_list, lidar_pos)
            occlusion_ratio, occlusion_state  = fast_occlusion_detection(vehicle, other_vehicles_list, lidar_pos)
            
            # 保存遮挡状态 (直接使用原生Python类型)
            vehicle['occluded_state'] = int(occlusion_state)
            vehicle['occlusion_ratio'] = float(round(occlusion_ratio, 4))
        
        # 保存修改后的数据
        output_path = os.path.join(agent_id_path, 
                                    timestamp.replace('.yaml', '_occluded_state.yaml'))
                    # 转换numpy数组和其他不可序列化对象
        try:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, 
                         default_flow_style=False,  # 使用块格式
                         sort_keys=False,           # 保持键的顺序
                         width=1000)               # 避免长行折行
                print(f"保存文件 {output_path} 成功")
        except Exception as e:
            print(f"保存文件 {output_path} 时出错: {e}")

def main():
    """主函数"""
    # 指定数据集路径
    V2XSET_PATH = '/home/node/code/zzh/HEAL/dataset/V2XSET'
    OPV2V_PATH = '/home/zzh/projects/HEAL/dataset/OPV2V'
    path = OPV2V_PATH
    splits = ['train', 'test', 'validate']
    
    for split in splits:
        split_path = os.path.join(path, split)
        print(f'正在处理 {split} 集...')
        
        # 遍历所有场景日期
        for scene_date in tqdm(os.listdir(split_path)):
            date_path = os.path.join(split_path, scene_date)
            if not os.path.isdir(date_path):
                continue
                
            # 遍历所有代理ID
            for agent_id in os.listdir(date_path):
                agent_id_path = os.path.join(date_path, agent_id)
                if not os.path.isdir(agent_id_path):
                    continue
                    
                process_scene(agent_id_path)
from opencood.visualization.simple_plot3d.canvas_bev import Canvas_BEV
import matplotlib.pyplot as plt
import opencood.utils.pcd_utils as pcd_utils
def vis_agnt_time_label(path = '/home/node/code/zzh/HEAL/dataset/V2XSET/train/2021_08_18_19_11_02', agent = '3242',time = '000080'):
    
    "可视化每个agent在某时间戳下的yaml的目标"
    
    
    yaml_path = os.path.join(path, agent, time + '.yaml')
    data = load_yaml(yaml_path)
    vehicles = data.get('vehicles', [])
    if not vehicles:
        print("没有车辆数据")
        return
    # 获取ego的lidar pose
    lidar_pose = data.get('lidar_pose')
    if lidar_pose is None:
        print("没有lidar_pose数据")
        return
    

    
    def get_transform_matrix(pose):
        """从pose [x,y,z,roll,pitch,yaw]构建4x4变换矩阵"""
        x, y, z = pose[:3]
        roll, pitch, yaw = pose[3:]
        yaw = -np.radians(yaw)
        # 先构建旋转矩阵
        cos_roll = np.cos(np.radians(roll))
        sin_roll = np.sin(np.radians(roll))
        cos_pitch = np.cos(np.radians(pitch))
        sin_pitch = np.sin(np.radians(pitch))
        cos_yaw = np.cos(np.radians(yaw))
        sin_yaw = np.sin(np.radians(yaw))
        
        # 构建旋转矩阵 (roll -> pitch -> yaw)
        R_roll = np.array([[1, 0, 0], 
                          [0, cos_roll, -sin_roll],
                          [0, sin_roll, cos_roll]])
        
        R_pitch = np.array([[cos_pitch, 0, sin_pitch],
                           [0, 1, 0],
                           [-sin_pitch, 0, cos_pitch]])
        
        R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                         [sin_yaw, cos_yaw, 0],
                         [0, 0, 1]])
        
        R = R_yaw @ R_pitch @ R_roll
        
        # 构建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = [x, y, z]
        
        return transform
    # 计算world到ego的变换矩阵
    world_to_ego = np.linalg.inv(get_transform_matrix(lidar_pose))
    # # 增加一个额外的X轴反转
    # flip_x = np.eye(4)
    # flip_x[2, 2] = -1  # X轴反转
    # world_to_ego = world_to_ego @ flip_x  # 在变换后应用X轴反转
    # 加载点云文件
    pcd_path = yaml_path.replace('.yaml', '.pcd')
    if not os.path.exists(pcd_path):
        print(f"点云文件不存在: {pcd_path}")
        return
    # 创建BEV画布
    canvas_bev = Canvas_BEV(canvas_shape=(400, 400),
                          canvas_x_range=(-100, 100),
                          canvas_y_range=(-100, 100),
                          left_hand=True)
    
    # 1. 绘制点云
    def flip_x_coordinates(points_or_boxes):
        flipped = points_or_boxes.copy()
        flipped[..., 0] = -flipped[..., 0]  # 翻转X坐标
        return flipped

    pcd_np = pcd_utils.pcd_to_np(pcd_path)
    # pcd_np_flipped = flip_x_coordinates(pcd_np)
    canvas_xy, valid_mask = canvas_bev.get_canvas_coords(pcd_np)
    canvas_bev.draw_canvas_points(canvas_xy[valid_mask])
    
    # 2. 绘制车辆边界框
    vehicles = data.get('vehicles', {})
    if not vehicles:
        print("没有车辆数据")
        return
        
    boxes = []
    def get_box_corners_vis(vehicle):
        """将车辆参数转换为8个角点坐标
        
        Args:
            vehicle (dict): 包含'location'、'center'、'extent'、'angle'的车辆信息
            
        Returns:
            np.ndarray: 形状为(8, 3)的角点坐标数组，顺序为：
            [前左下, 前右下, 后右下, 后左下, 前左上, 前右上, 后右上, 后左上]
        """
        # 获取基本参数
        loc = vehicle['location']                      # [x, y, z]
        center = vehicle.get('center', [0, 0, 0])      # 可选的偏移量
        extent = vehicle['extent']                     # [length/2, width/2, height/2]
        angle = vehicle['angle']                       # [roll, pitch, yaw]
        
         # 在局部坐标系中创建盒子角点
        l, w, h = extent                              # 半长度
        corners = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])
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
    boxes_3d = []  # 用于存储3D边界框格式 [x,y,z,l,w,h,yaw]
    for vehicle_id, vehicle in vehicles.items():
        
        # if not all(k in vehicle for k in ['location', 'angle', 'extent']):
        #     print(f"车辆 {vehicle_id} 缺少必要信息")
        #     continue
            
        # 使用 get_box_corners 获取8个顶点
        corners_world = get_box_corners_vis(vehicle)  # shape: (8, 3)
        # 转换到ego坐标系
        corners_homo = np.concatenate([corners_world, np.ones((8, 1))], axis=1)  # 转换为齐次坐标
        corners_ego = (world_to_ego @ corners_homo.T).T  # 转换到ego坐标系
        corners_ego = corners_ego[:, :3]  # 去掉齐次坐标的1
        boxes.append(corners_ego)

    if boxes:
        # boxes_flipped = []
        # for box in boxes:
        #     boxes_flipped.append(flip_x_coordinates(box))
        boxes = np.stack(boxes)
        print('boxes shape:', boxes.shape)
        for i in range(len(boxes)):
            boxes[i, :, 1] = -boxes[i, :, 1]  # 翻转每个边界框的X坐标
        # 绘制边界框
        canvas_bev.draw_boxes(boxes,colors=(0,255,0), texts=['']*len(boxes), box_line_thickness=1)
    
    # # 添加lidar位置标记
    # if 'lidar_pose' in data:
    #     lidar_pos = data['lidar_pose'][:3]
    #     canvas_bev.draw_ego_car(lidar_pos)
    
    # 保存可视化结果
    vis_save_path = yaml_path.replace('.yaml', '_vis.png')
    plt.imshow(canvas_bev.canvas)
    # plt.imshow(np.fliplr(canvas_bev.canvas))
    plt.savefig(vis_save_path)
    plt.close()
    print(f"可视化结果已保存到: {vis_save_path}")
    
if __name__ == "__main__":
    main()
    # path = '/home/node/code/zzh/HEAL/dataset/V2XSET/train/2021_08_23_13_10_47/7694/000070_occluded_state.yaml'
    # path = '/home/node/code/zzh/HEAL/dataset/V2XSET/train/2021_08_24_09_25_42/12954/000112'
    # vis_agnt_time_label('/home/node/code/zzh/HEAL/dataset/V2XSET/train/2021_08_24_09_25_42', '12954', '000112')