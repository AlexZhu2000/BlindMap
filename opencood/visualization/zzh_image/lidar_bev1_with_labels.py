import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from matplotlib.patches import Polygon

# Try to import Open3D, but provide fallback if it fails
try:
    import open3d as o3d
    has_open3d = True
except ImportError:
    has_open3d = False
    print("Warning: Open3D not available. Using limited functionality.")

def load_pcd_file(pcd_file_path, pc_range):
    """
    Load point cloud data from file.
    Supports both Open3D's PCD format and NumPy arrays.
    """
    if not os.path.exists(pcd_file_path):
        print(f"文件不存在: {pcd_file_path}")
        return None
    
    # Check if it's a binary PCD file (basic check)
    try:
        with open(pcd_file_path, 'rb') as f:
            header = f.read(1024).decode('utf-8', errors='ignore')
            if 'PCD' in header:
                if has_open3d:
                    try:
                        pcd = o3d.io.read_point_cloud(pcd_file_path)
                        pcd_np = np.asarray(pcd.points)
                        x = pcd_np[:, 0]
                        y = pcd_np[:, 1]
                        # Get valid mask
                        valid_mask = ((x > pc_range[0]) & 
                                    (x < pc_range[3]) &
                                    (y > pc_range[1]) & 
                                    (y < pc_range[4]))
                        print(valid_mask.shape)
                        print(f"pcd_np.shape: {pcd_np.shape}")
                        pcd_np = pcd_np[valid_mask]
                        print(f"成功加载PCD文件: {pcd_np.shape} points")
                        return pcd_np
                    except Exception as e:
                        print(f"Open3D加载PCD文件失败: {e}")
                        # Try alternative loading method below
                else:
                    print("需要Open3D库来读取PCD文件")
                    return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
    
    # Alternative: Try loading as NumPy array or CSV
    try:
        # Try loading as NumPy array
        data = np.loadtxt(pcd_file_path, delimiter=',', skiprows=1)
        return data
    except:
        try:
            # Try loading as space-separated values
            data = np.loadtxt(pcd_file_path, skiprows=11)  # Skip PCD header
            return data[:, 0:3]  # Assuming XYZ are the first 3 columns
        except Exception as e:
            print(f"所有加载方法均失败: {e}")
            return None

def load_json_boxes(json_file_path):
    """
    Load 3D bounding box information from JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"标注文件不存在: {json_file_path}")
        return None
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        boxes = []
        
        # Case 1: Direct list of objects
        if isinstance(data, list):
            boxes = data
        # Case 2: Single object
        elif isinstance(data, dict) and '3d_location' in data:
            boxes = [data]
        # Case 3: Dictionary with 'objects' key
        elif isinstance(data, dict) and 'objects' in data:
            boxes = data['objects']
        # Case 4: Nested structure, search for objects
        else:
            # Try to find arrays that might contain box data
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if items look like 3D boxes
                    if isinstance(value[0], dict) and ('3d_dimensions' in value[0] or '3d_location' in value[0]):
                        boxes = value
                        break
        
        if not boxes:
            print("警告: 未在JSON中找到边界框数据，请检查格式")
            
        # Debug: Print first box to verify format
        if boxes and len(boxes) > 0:
            print("示例边界框数据:", json.dumps(boxes[0], indent=2))
            
        return boxes
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return None

def draw_rotated_box(ax, center_x, center_y, length, width, rotation, color='r', linewidth=2, alpha=1.0, zorder=20):
    """
    Draw a rotated 2D bounding box (top view of 3D box) with simplified styling.
    Just draws the box outline similar to the example image.
    """
    # Convert rotation to radians if not already
    if abs(rotation) < np.pi:  # Assuming rotation is in radians if small
        theta = rotation
    else:
        theta = np.radians(rotation)
    
    # Calculate the four corners of the box
    corners = []
    for i, j in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        # Use length for X and width for Y (automotive convention)
        x = center_x + (j * length/2 * np.cos(theta) - i * width/2 * np.sin(theta))
        y = center_y + (j * length/2 * np.sin(theta) + i * width/2 * np.cos(theta))
        corners.append((x, y))
    
    # Draw the box as a simple outline
    poly = Polygon(corners, closed=True, edgecolor=color, facecolor='none', 
                  fill=False, linewidth=linewidth, zorder=zorder)
    ax.add_patch(poly)
    
    # Draw direction indicator (front of vehicle)
    front_x = center_x + length/2 * np.cos(theta)
    front_y = center_y + length/2 * np.sin(theta)
    ax.plot([center_x, front_x], [center_y, front_y], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

def visualize_point_cloud_top_view(pcd_file_path=None, json_file_path=None, save_path="top_view.png", 
                                  pc_range=[-140, -40, -3, 140, 40, 1],
                                  point_size=0.5, dpi=300, figsize=(2800, 800),
                                  show_axes=True, density_heatmap=False):
    """
    从点云数据创建俯视图并保存为图像，同时显示3D边界框
    
    参数:
        pcd_file_path: 点云文件路径
        json_file_path: 包含3D边界框的JSON文件路径
        save_path: 输出图像路径
        point_size: 点的大小
        dpi: 输出图像DPI
        figsize: 图像尺寸
        show_axes: 是否显示坐标轴
        density_heatmap: 是否创建密度热图代替散点图
    """
    # 加载或创建点云数据
    if pcd_file_path is None:
        print("未提供点云文件...")
        return 0
    else:
        print(f"加载点云文件: {pcd_file_path}")
        points = load_pcd_file(pcd_file_path, pc_range)
        
        if points is None or len(points) == 0:
            print("无法加载有效点云...")
            return 0
    
    # 加载3D边界框数据
    boxes = None
    if json_file_path is not None:
        print(f"加载3D边界框文件: {json_file_path}")
        boxes = load_json_boxes(json_file_path)
        if boxes is None or len(boxes) == 0:
            print("无法加载有效3D边界框数据...")
    
    # 创建俯视图 (XY平面)
    try:
        figsize = ((pc_range[3] - pc_range[0])/10, (pc_range[4] - pc_range[1])/10)
        fig, ax = plt.subplots(figsize=figsize)
        
        if density_heatmap and len(points) > 100:
            # 创建热图以显示点云密度
            h = ax.hist2d(points[:, 0], points[:, 1], bins=100, cmap='viridis', zorder=1)
            plt.colorbar(h[3], ax=ax, label='点密度')
            plt.title('点云俯视图 (密度热图) 与3D边界框')
        else:
            # 创建散点图，确保点云在边界框下方
            ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.6, zorder=1)
            plt.title('点云俯视图 与3D边界框')
        
        # 绘制3D边界框（俯视图）
        # When drawing boxes in both standard and height-colored views
        if boxes is not None and len(boxes) > 0:
            print(f"绘制 {len(boxes)} 个3D边界框...")
            for i, box in enumerate(boxes):
                try:
                    # Extract box information
                    if '3d_location' in box and isinstance(box['3d_location'], dict):
                        center_x = box['3d_location'].get('x', 0)
                        center_y = box['3d_location'].get('y', 0)
                    else:
                        continue
                        
                    if '3d_dimensions' in box and isinstance(box['3d_dimensions'], dict):
                        length = box['3d_dimensions'].get('l', 0)
                        width = box['3d_dimensions'].get('w', 0)
                    else:
                        continue
                        
                    rotation = box.get('rotation', 0)
                    occluded_state = box.get('occluded_state', 0)
                    
                    # Set color based on occlusion state
                    color = 'lime' if occluded_state == 0 else 'red'
                    
                    # Use simplified drawing function without labels
                    draw_rotated_box(ax, center_x, center_y, length, width, rotation, color=color)
                    
                    # Remove the text label code here
                except Exception as e:
                    print(f"绘制边界框时出错: {e}")
        
        if show_axes:
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.axis('off')
        
        ax.set_aspect('equal')  # 保持XY比例一致
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"俯视图已保存至: {save_path}")
        plt.close()
        
        # 额外创建一个包含高度信息的彩色俯视图
        if len(points) > 0 and points.shape[1] >= 3:
            fig, ax = plt.subplots(figsize=figsize)
            
            # 根据Z值着色，设置zorder确保点云在边界框下方
            scatter = ax.scatter(points[:, 0], points[:, 1], s=point_size, 
                              c=points[:, 2], cmap='viridis', alpha=0.8, zorder=1)
            plt.colorbar(scatter, ax=ax, label='高度 (Z)')
            plt.title('点云俯视图 (高度着色) 与3D边界框')
            
            # 在高度着色图上也绘制3D边界框
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        if '3d_location' in box and isinstance(box['3d_location'], dict):
                            center_x = box['3d_location'].get('x', 0)
                            center_y = box['3d_location'].get('y', 0)
                        else:
                            continue
                            
                        if '3d_dimensions' in box and isinstance(box['3d_dimensions'], dict):
                            length = box['3d_dimensions'].get('l', 0)
                            width = box['3d_dimensions'].get('w', 0)
                        else:
                            continue
                            
                        rotation = box.get('rotation', 0)
                        occluded_state = box.get('occluded_state', 0)
                        
                        color = 'lime' if occluded_state == 0 else 'red'
                        draw_rotated_box(ax, center_x, center_y, length, width, rotation, color=color)
                        
                        # 添加对象类型标签
                        obj_type = box.get('type', '')
                        if obj_type:
                            ax.text(center_x, center_y, obj_type, fontsize=12, 
                                   color='black', weight='bold', ha='center', va='center', 
                                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'), 
                                   zorder=30)
                    except Exception as e:
                        print(f"在高度着色图上绘制边界框时出错: {e}")
            
            if show_axes:
                ax.set_xlabel('X', fontsize=12)
                ax.set_ylabel('Y', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.axis('off')
                
            ax.set_aspect('equal')
            ax.set_xlim(pc_range[0], pc_range[3])
            ax.set_ylim(pc_range[1], pc_range[4])
            
            height_colored_path = save_path.replace('.png', '_height_colored.png')
            plt.savefig(height_colored_path, bbox_inches='tight', dpi=dpi)
            print(f"高度着色俯视图已保存至: {height_colored_path}")
            plt.close()
            
    except Exception as e:
        print(f"创建可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存原始点云数据作为最后的备选
        try:
            np.savetxt(save_path.replace('.png', '.csv'), 
                      points[:, 0:min(points.shape[1], 3)],
                      delimiter=',', 
                      header='X,Y,Z')
            print(f"无法生成图像，点云坐标已保存至: {save_path.replace('.png', '.csv')}")
        except Exception as e:
            print(f"保存CSV也失败: {e}")

if __name__ == "__main__":
    # 处理命令行参数
    pcd_path = None
    json_path = None
    id = "004364"
    save_path = os.path.join("/home/zzh/projects/Where2comm/opencood/visualization/zzh_image/", f"pcd_bev_with_labels_back_up{id}.png")
    
    if len(sys.argv) > 1:
        pcd_path = sys.argv[1]
    if len(sys.argv) > 2:
        json_path = sys.argv[2]
    if len(sys.argv) > 3:
        save_path = sys.argv[3]
    
    # 使用默认值（如果未指定）
    if pcd_path is None:
        pcd_path = f'/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/velodyne/{id}.pcd'
    if json_path is None:
        json_path = f'/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/label/lidar_backup/{id}.json'
    pc_range = [-140, -40, -3, 140, 40, 1]
    # 创建可视化
    visualize_point_cloud_top_view(
        pcd_file_path=pcd_path,
        json_file_path=json_path,
        save_path=save_path,
        pc_range=pc_range,
        point_size=0.5,  # 较小的点以显示更多细节
        density_heatmap=False  # 使用热图来更好地显示密集区域
    )