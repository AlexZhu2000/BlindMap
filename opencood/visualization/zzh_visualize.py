import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def check_open3d_visualization():
    """
    检查Open3D可视化功能是否正常工作
    返回True表示可用，False表示不可用
    """
    try:
        # 创建一个简单的几何体
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh.compute_vertex_normals()
        
        # 尝试创建一个可视化窗口
        vis = o3d.visualization.Visualizer()
        result = vis.create_window(visible=True, width=10, height=10, window_name="Test")
        
        if not result:
            print("无法创建Open3D窗口")
            return False
            
        # 添加几何体
        vis.add_geometry(mesh)
        
        # 捕获一个帧来验证渲染
        vis.poll_events()
        vis.update_renderer()
        
        # 测试截图功能
        image = vis.capture_screen_float_buffer(do_render=True)
        if image is None:
            print("无法捕获屏幕截图")
            return False
            
        # 关闭窗口
        vis.destroy_window()
        
        print("Open3D 可视化功能工作正常！")
        return True
    except Exception as e:
        print(f"Open3D 可视化测试失败: {e}")
        return False

def create_bbox_from_label(label):
    """
    从标签创建3D边界框
    """
    # 提取3D边界框信息
    h = label["3d_dimensions"]["h"]
    w = label["3d_dimensions"]["w"]
    l = label["3d_dimensions"]["l"]
    x = label["3d_location"]["x"]
    y = label["3d_location"]["y"]
    z = label["3d_location"]["z"]
    rotation = label["rotation"]
    
    # 创建边界框
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = np.array([x, y, z])
    bbox.extent = np.array([l, w, h])  # 注意：Open3D的顺序是[长, 宽, 高]
    
    # 设置旋转（绕Y轴旋转）
    R = np.array([
        [np.cos(rotation), 0, np.sin(rotation)],
        [0, 1, 0],
        [-np.sin(rotation), 0, np.cos(rotation)]
    ])
    bbox.R = R
    
    # 设置颜色（根据类型）
    if label["type"].lower() == "car":
        bbox.color = np.array([1, 0, 0])  # 红色
    elif label["type"].lower() == "pedestrian":
        bbox.color = np.array([0, 1, 0])  # 绿色
    elif label["type"].lower() == "cyclist":
        bbox.color = np.array([0, 0, 1])  # 蓝色
    else:
        bbox.color = np.array([1, 1, 0])  # 黄色
        
    return bbox

def visualize_point_cloud_with_labels(pcd_file_path, json_file_path, save_path="visualization.png", debug_mode=False):
    """
    加载点云和标签，并一起可视化
    
    参数:
        pcd_file_path: 点云文件路径
        json_file_path: 包含目标标签的JSON文件路径
        save_path: 保存可视化结果的图像路径
        debug_mode: 是否启用调试模式
    """
    # 检查Open3D可视化是否可用
    if debug_mode:
        open3d_vis_available = check_open3d_visualization()
        print(f"Open3D 可视化可用性: {open3d_vis_available}")
    else:
        open3d_vis_available = True
    
    # 加载点云
    try:
        print(f"加载点云: {pcd_file_path}")
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        if len(pcd.points) == 0:
            raise ValueError("点云为空")
    except Exception as e:
        print(f"加载点云失败: {e}")
        print("创建示例点云...")
        pcd = o3d.geometry.PointCloud()
        points = np.random.rand(1000, 3) * 30 - 15  # 创建范围在-15到15的随机点
        pcd.points = o3d.utility.Vector3dVector(points)
    
    # 读取JSON文件中的标签
    try:
        print(f"读取标签: {json_file_path}")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 处理不同格式的JSON
        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict) and "objects" in data:
            labels = data["objects"]
        else:
            # 尝试提取第一层可能的标签列表
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    labels = value
                    break
            else:
                raise ValueError("无法识别的JSON格式")
    except Exception as e:
        print(f"读取标签失败: {e}")
        # 创建示例标签
        labels = [{
            "type": "Car",
            "truncated_state": "0",
            "occluded_state": "0",
            "alpha": "0",
            "2d_box": {
                "xmin": 98.036835,
                "ymin": 696.638428,
                "xmax": 355.456085,
                "ymax": 918.447144
            },
            "3d_dimensions": {
                "h": 1.446576,
                "w": 1.970789,
                "l": 4.359478
            },
            "3d_location": {
                "x": 20.781150589759154,
                "y": 0.01655674367897273,
                "z": -0.8951699247385125
            },
            "rotation": 0.0245684483504741
        }]
    
    # 创建3D边界框
    bboxes = []
    for label in labels:
        try:
            bbox = create_bbox_from_label(label)
            bboxes.append(bbox)
        except Exception as e:
            print(f"创建标签的边界框失败: {e}")
    
    # 使用Open3D可视化（如果可用）
    if open3d_vis_available:
        try:
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Point Cloud with Labels", width=1280, height=720)
            
            # 添加点云
            vis.add_geometry(pcd)
            
            # 添加所有边界框
            for bbox in bboxes:
                # 创建线框表示
                bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
                bbox_lines.colors = o3d.utility.Vector3dVector([bbox.color for _ in range(12)])
                vis.add_geometry(bbox_lines)
            
            # 设置渲染选项
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            opt.point_size = 1.0
            
            # 设置相机位置为俯视图
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])  # 相机朝向 -z 方向
            ctr.set_up([0, 1, 0])      # 上方向为 +y
            ctr.set_lookat([0, 0, 0])  # 看向原点
            ctr.set_zoom(0.5)          # 设置缩放
            
            # 更新渲染
            vis.update_geometry(pcd)
            for bbox in bboxes:
                vis.update_geometry(bbox)
            
            vis.poll_events()
            vis.update_renderer()
            
            # 捕获截图并保存
            image = vis.capture_screen_float_buffer(True)
            plt.figure(figsize=(12, 8))
            plt.imshow(np.asarray(image))
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"已保存可视化图像至: {save_path}")
            
            # 在显示窗口前，创建一个带有图例的版本
            legend_save_path = os.path.splitext(save_path)[0] + "_with_legend.png"
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(np.asarray(image))
            ax.axis('off')
            
            # 添加图例
            class_colors = {
                "Car": "red",
                "Pedestrian": "green", 
                "Cyclist": "blue",
                "Other": "yellow"
            }
            
            # 创建图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=class_name) 
                              for class_name, color in class_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # 添加统计信息
            class_counts = {}
            for label in labels:
                class_name = label["type"]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            stats_text = "目标统计:\n"
            for class_name, count in class_counts.items():
                stats_text += f"{class_name}: {count}\n"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.savefig(legend_save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"已保存带图例的可视化图像至: {legend_save_path}")
            
            # 如果要显示交互窗口，取消注释下面的行
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Open3D可视化失败: {e}")
            open3d_vis_available = False
    
    # 如果Open3D可视化不可用，使用matplotlib
    if not open3d_vis_available:
        print("使用Matplotlib备选方案进行可视化...")
        
        # 使用matplotlib进行3D可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云（限制点的数量以提高性能）
        points = np.asarray(pcd.points)
        max_points = min(10000, len(points))  # 最多显示10000个点
        indices = np.random.choice(len(points), max_points, replace=False)
        ax.scatter(points[indices, 0], points[indices, 1], points[indices, 2], 
                  s=1, c='lightgray', alpha=0.5)
        
        # 绘制边界框
        for label in labels:
            try:
                # 获取3D边界框参数
                h = label["3d_dimensions"]["h"]
                w = label["3d_dimensions"]["w"]
                l = label["3d_dimensions"]["l"]
                x = label["3d_location"]["x"]
                y = label["3d_location"]["y"]
                z = label["3d_location"]["z"]
                rotation = label["rotation"]
                
                # 设置颜色
                if label["type"].lower() == "car":
                    color = 'red'
                elif label["type"].lower() == "pedestrian":
                    color = 'green'
                elif label["type"].lower() == "cyclist":
                    color = 'blue'
                else:
                    color = 'yellow'
                
                # 创建边界框的8个顶点
                # 先创建一个标准的未旋转的边界框
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
                corners = np.vstack([x_corners, y_corners, z_corners])
                
                # 应用旋转（绕Y轴）
                R = np.array([
                    [np.cos(rotation), 0, np.sin(rotation)],
                    [0, 1, 0],
                    [-np.sin(rotation), 0, np.cos(rotation)]
                ])
                corners = np.dot(R, corners)
                
                # 移动到目标位置
                corners[0, :] += x
                corners[1, :] += y
                corners[2, :] += z
                
                # 绘制边界框的12条边
                def draw_line(p1, p2):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)
                
                # 下面四条边
                draw_line(corners[:, 0], corners[:, 1])
                draw_line(corners[:, 1], corners[:, 2])
                draw_line(corners[:, 2], corners[:, 3])
                draw_line(corners[:, 3], corners[:, 0])
                
                # 上面四条边
                draw_line(corners[:, 4], corners[:, 5])
                draw_line(corners[:, 5], corners[:, 6])
                draw_line(corners[:, 6], corners[:, 7])
                draw_line(corners[:, 7], corners[:, 4])
                
                # 连接上下的四条边
                draw_line(corners[:, 0], corners[:, 4])
                draw_line(corners[:, 1], corners[:, 5])
                draw_line(corners[:, 2], corners[:, 6])
                draw_line(corners[:, 3], corners[:, 7])
                
                # 标注类型
                # ax.text(x, y, z+h/2, label["type"], color=color)
                ax.text(x, y, z+h/2, color=color)
                
            except Exception as e:
                print(f"绘制边界框失败: {e}")
        
        # 设置图表属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('点云与3D边界框')
        
        # 设置为俯视角度
        ax.view_init(elev=90, azim=-90)
        
        # 保存图像
        # plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # 创建一个带有图例的版本
        legend_save_path = os.path.splitext(save_path)[0] + "_with_legend.png"
        
        # 添加图例
        class_colors = {
            "Car": "red",
            "Pedestrian": "green", 
            "Cyclist": "blue",
            "Other": "yellow"
        }
        
        # # 创建图例
        # from matplotlib.patches import Patch
        # legend_elements = [Patch(facecolor=color, label=class_name) 
        #                   for class_name, color in class_colors.items()]
        # ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加统计信息
        class_counts = {}
        for label in labels:
            class_name = label["type"]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        stats_text = "目标统计:\n"
        for class_name, count in class_counts.items():
            stats_text += f"{class_name}: {count}\n"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.savefig(legend_save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存可视化图像至: {save_path}")
        print(f"已保存带图例的可视化图像至: {legend_save_path}")

def print_opengl_info():
    """打印OpenGL相关环境信息"""
    print("\n===== OpenGL/GLFW 诊断信息 =====")
    
    # 检查显示环境变量
    print(f"DISPLAY环境变量: {os.environ.get('DISPLAY', 'Not set')}")
    
    # 检查OpenGL相关环境变量
    gl_vars = [key for key in os.environ.keys() if 'GL' in key]
    if gl_vars:
        print("OpenGL相关环境变量:")
        for var in gl_vars:
            print(f"  {var} = {os.environ[var]}")
    else:
        print("未设置OpenGL相关环境变量")
    
    # 尝试获取Open3D版本
    try:
        print(f"Open3D版本: {o3d.__version__}")
    except:
        print("无法获取Open3D版本")
    
    # 尝试测试GLFW
    try:
        import glfw
        initialized = glfw.init()
        print(f"GLFW初始化: {'成功' if initialized else '失败'}")
        if initialized:
            print(f"GLFW版本: {glfw.get_version_string()}")
            
            # 尝试创建一个窗口
            try:
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # 创建不可见窗口
                window = glfw.create_window(100, 100, "Test", None, None)
                print(f"GLFW创建窗口: {'成功' if window else '失败'}")
                if window:
                    glfw.destroy_window(window)
            except Exception as e:
                print(f"GLFW创建窗口异常: {e}")
                
            glfw.terminate()
    except ImportError:
        print("无法导入GLFW库")
    except Exception as e:
        print(f"GLFW测试异常: {e}")
    
    # 尝试执行简单的X11命令
    try:
        import subprocess
        result = subprocess.run(['xdpyinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("X11服务器可用")
            # 提取一些有用信息
            vendor_line = [line for line in result.stdout.split('\n') if 'vendor string' in line]
            if vendor_line:
                print(f"  {vendor_line[0].strip()}")
            
            extensions = [line for line in result.stdout.split('\n') if 'GLX' in line and 'extensions' in line]
            if extensions:
                print(f"  {extensions[0].strip()}")
        else:
            print(f"X11服务器不可用: {result.stderr}")
    except Exception as e:
        print(f"X11测试异常: {e}")
    
    print("================================\n")

import math
def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T


def read_label_bboxes(label_path):
    with open(label_path, "r") as load_f:
        labels = json.load(load_f)

    boxes = []
    for label in labels:
        obj_size = [
            float(label["3d_dimensions"]["l"]),
            float(label["3d_dimensions"]["w"]),
            float(label["3d_dimensions"]["h"]),
        ]
        # print(label['occluded_state'])
        yaw_lidar = float(label["rotation"])
        center_lidar = [
            float(label["3d_location"]["x"]),
            float(label["3d_location"]["y"]),
            float(label["3d_location"]["z"]),
        ]

        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        boxes.append(box)

    return boxes
import simple_plot3d.canvas_bev as canvas_bev
def visulaize_pcd_label_BEV(pcd_path, label_path,save_path,  pc_range = [-140, -40, -3, 140, 40, 1], left_hand=False):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_np = np.asarray(pcd.points)
    bboxes = read_label_bboxes(label_path)
    canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                        canvas_x_range=(pc_range[0], pc_range[3]), 
                                        canvas_y_range=(pc_range[1], pc_range[4]),
                                        left_hand=left_hand
                                        ) 

    canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
    canvas.draw_canvas_points(canvas_xy[valid_mask])
    # color_list = [(0, 206, 209),(255, 215,0)]
    # for i, pcd_np_t in enumerate(pcd_np[1:2]):
    #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np_t) # Get Canvas Coords
    #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[i-1]) # Only draw valid points
    box_line_thickness = 5
    
    # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
    canvas.draw_boxes(np.array(bboxes),colors=(0,255,0), texts=None, box_line_thickness=box_line_thickness)
    
    # if vis_pred_box:
    #     canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name, box_line_thickness=box_line_thickness)


    plt.axis("off")

    # plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
    plt.clf()
if __name__ == "__main__":
    # 处理命令行参数
    pcd_path = r'/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/velodyne/000289.pcd'
    label_path = r'/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/label/lidar/000289.json'
    save_path = r'/home/zzh/projects/Where2comm/opencood/visualization/zzh_image/000289.png'
    visulaize_pcd_label_BEV(pcd_path, label_path,save_path)
    # import argparse
    # parser = argparse.ArgumentParser(description='点云与3D边界框可视化')
    # parser.add_argument('--pcd', type=str, help='点云文件路径', default=pcd_path)
    # parser.add_argument('--json', type=str, help='标签JSON文件路径,', default=label_path)
    # parser.add_argument('--output', type=str, default='visualization.png', help='输出图像路径')
    # parser.add_argument('--debug', action='store_true', help='启用调试模式', default=True)
    # parser.add_argument('--info', action='store_true', help='打印OpenGL和环境信息', default=True)
    
    # args = parser.parse_args()
    
    # # 如果请求打印信息
    # if args.info:
    #     print_opengl_info()
    
    # # 如果提供了点云和JSON文件路径，则进行可视化
    # if args.pcd and args.json:
    #     visualize_point_cloud_with_labels(args.pcd, args.json, args.output, args.debug)
    # elif args.debug:
    #     # 只进行Open3D可视化测试
    #     check_open3d_visualization()
    # else:
    #     print("请提供点云文件和JSON标签文件的路径")
    #     print("使用方法: python script.py --pcd point_cloud.pcd --json labels.json --output vis.png")
    #     print("          python script.py --debug  # 测试Open3D可视化功能")
    #     print("          python script.py --info   # 打印OpenGL和环境信息")

    