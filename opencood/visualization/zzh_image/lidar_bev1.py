import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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

def visualize_point_cloud_top_view(pcd_file_path=None, save_path="top_view.png", pc_range=[-140, -40, -3, 140, 40, 1],
                                  point_size=0.5, dpi=300, figsize=(2800, 800),
                                  show_axes=True, density_heatmap=False):
    """
    从点云数据创建俯视图并保存为图像
    
    参数:
        pcd_file_path: 点云文件路径
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
    
    # 创建俯视图 (XY平面)
    try:
        figsize = ((pc_range[3] - pc_range[0])/10, (pc_range[4] - pc_range[1])/10)
        plt.figure(figsize=figsize)
        
        if density_heatmap and len(points) > 100:
            # 创建热图以显示点云密度
            plt.hist2d(points[:, 0], points[:, 1], bins=100, cmap='viridis')
            plt.colorbar(label='点密度')
            plt.title('点云俯视图 (密度热图)')
        else:
            # 创建散点图
            plt.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.6)
            plt.title('点云俯视图')
        
        if show_axes:
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.axis('off')
        
        plt.axis('equal')  # 保持XY比例一致
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"俯视图已保存至: {save_path}")
        plt.close()
        
        # 额外创建一个包含高度信息的彩色俯视图
        if len(points) > 0 and points.shape[1] >= 3:
            plt.figure(figsize=figsize)
            # 根据Z值着色
            scatter = plt.scatter(points[:, 0], points[:, 1], s=point_size, c=points[:, 2], cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, label='高度 (Z)')
            plt.title('点云俯视图 (高度着色)')
            
            if show_axes:
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.axis('off')
                
            plt.axis('equal')
            height_colored_path = save_path.replace('.png', '_height_colored.png')
            plt.savefig(height_colored_path, bbox_inches='tight', dpi=dpi)
            print(f"高度着色俯视图已保存至: {height_colored_path}")
            plt.close()
            
    except Exception as e:
        print(f"创建可视化时出错: {e}")
        
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
    save_path = "/home/zzh/projects/Where2comm/opencood/visualization/zzh_image/point_cloud_top_view.png"
    
    if len(sys.argv) > 1:
        pcd_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    
    # 使用默认值（如果未指定）
    if pcd_path is None:
        pcd_path = r'/home/zzh/projects/Where2comm/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/velodyne/000289.pcd'
    pc_range = [-140, -40, -3, 140, 40, 1]
    # 创建可视化
    visualize_point_cloud_top_view(
        pcd_file_path=pcd_path, 
        save_path=save_path,
        pc_range = pc_range,
        point_size=0.5,  # 较小的点以显示更多细节
        density_heatmap=False  # 使用热图来更好地显示密集区域
    )