"""
Dataset visualization script for BlindMap with occluded states.
This script loads the dataset using the standard OpenCOOD dataset loader
and visualizes ego vehicle, cooperative vehicles, and BlindMap using BEV canvas.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add OpenCOOD to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.utils.box_utils import boxes_to_corners_3d


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset with BlindMap")
    parser.add_argument("--hypes_yaml", type=str, 
                        default="/home/zzh/projects/BlindMap/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid_blindmap.yaml",
                        help='Config yaml file')
    parser.add_argument('--save_dir', type=str, 
                        default='/home/zzh/projects/BlindMap/opencood/visualization/vis_output',
                        help='Directory to save visualizations')
    parser.add_argument('--idx', type=int, default=0,
                        help='Index of the sample to visualize')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    return args


def visualize_bev_with_blindmap(batch_data, save_path, pc_range):
    """
    Visualize BEV point cloud with bounding boxes and BlindMap
    
    Args:
        batch_data: dict containing processed batch data
        save_path: path to save the visualization
        pc_range: lidar range [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    # Extract data from batch (first sample in batch)
    # Check available modality inputs
    ego_lidar = None
    for modality in ['m1', 'm2', 'm3', 'm4']:
        key = f'inputs_{modality}'
        if key in batch_data['ego'] and batch_data['ego'][key] is not None:
            # Get voxel features - need to convert back to point cloud or use origin_lidar
            if 'origin_lidar' in batch_data['ego']:
                ego_lidar = batch_data['ego']['origin_lidar'][0].cpu().numpy()  # [N, 4]
                break
    
    # If origin_lidar not available, skip this sample
    if ego_lidar is None:
        print("Warning: No lidar data available in batch")
        return
    
    # Get BlindMap
    ego_blind_map = batch_data['ego'].get('blind_map_ego', None)
    if ego_blind_map is not None:
        ego_blind_map = ego_blind_map[0].cpu().numpy()  # [H, W]
    
    # Get ground truth boxes
    object_bbx_center = batch_data['ego']['object_bbx_center']
    object_bbx_mask = batch_data['ego']['object_bbx_mask']
    
    # Convert to numpy and filter valid boxes
    if isinstance(object_bbx_center, torch.Tensor):
        object_bbx_center = object_bbx_center.cpu().numpy()
        object_bbx_mask = object_bbx_mask.cpu().numpy()
    
    # Get valid boxes (first sample in batch)
    valid_mask = object_bbx_mask[0] > 0
    valid_boxes = object_bbx_center[0][valid_mask]  # [N, 7] -> (x, y, z, dx, dy, dz, yaw)
    
    # Convert boxes to corners for visualization
    if len(valid_boxes) > 0:
        # boxes_to_corners_3d expects [N, 7] format
        gt_boxes_corners = boxes_to_corners_3d(valid_boxes, order='hwl')  # [N, 8, 3]
    else:
        gt_boxes_corners = np.zeros((0, 8, 3))
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    ## Subplot 1: Ego vehicle BEV
    ax1 = plt.subplot(1, 3, 1)
    canvas_ego = canvas_bev.Canvas_BEV_heading_right(
        canvas_shape=(int((pc_range[4]-pc_range[1])*10), int((pc_range[3]-pc_range[0])*10)),
        canvas_x_range=(pc_range[0], pc_range[3]), 
        canvas_y_range=(pc_range[1], pc_range[4]),
        left_hand=True
    )
    
    canvas_xy, valid_mask = canvas_ego.get_canvas_coords(ego_lidar)
    canvas_ego.draw_canvas_points(canvas_xy[valid_mask])
    canvas_ego.draw_boxes(gt_boxes_corners, colors=(0, 255, 0), texts=[''] * len(gt_boxes_corners))
    
    ax1.axis("off")
    ax1.imshow(canvas_ego.canvas)
    ax1.set_title("Ego Vehicle - BEV Point Cloud & GT Boxes", fontsize=14, pad=10)
    
    ## Subplot 2: All cooperative vehicles aggregated
    ax2 = plt.subplot(1, 3, 2)
    
    # Get lidar pose information to extract cooperative vehicles
    record_len = batch_data['ego']['record_len'][0].item()  # number of agents
    lidar_pose = batch_data['ego']['lidar_pose'].cpu().numpy()  # [N, 6]
    
    canvas_coop = canvas_bev.Canvas_BEV_heading_right(
        canvas_shape=(int((pc_range[4]-pc_range[1])*10), int((pc_range[3]-pc_range[0])*10)),
        canvas_x_range=(pc_range[0], pc_range[3]), 
        canvas_y_range=(pc_range[1], pc_range[4]),
        left_hand=True
    )
    
    # Draw ego point cloud in one color (blue)
    canvas_xy, valid_mask = canvas_coop.get_canvas_coords(ego_lidar)
    canvas_coop.draw_canvas_points(canvas_xy[valid_mask], colors=(100, 100, 255))
    
    # Draw cooperative vehicles in different color (orange)
    # In intermediate fusion, we only have aggregated origin_lidar
    # We show all points but could separate if needed
    if record_len > 1:
        # If we have origin_lidar with multiple agents, it's already aggregated
        # Just show in different color to indicate cooperation
        canvas_xy_coop, valid_mask_coop = canvas_coop.get_canvas_coords(ego_lidar)
        # Draw a subset in orange to show cooperation
        num_points = len(ego_lidar)
        points_per_agent = num_points // record_len
        if points_per_agent > 0:
            for i in range(1, record_len):
                start_idx = i * points_per_agent
                end_idx = (i + 1) * points_per_agent if i < record_len - 1 else num_points
                agent_points = ego_lidar[start_idx:end_idx]
                canvas_xy_agent, valid_mask_agent = canvas_coop.get_canvas_coords(agent_points)
                canvas_coop.draw_canvas_points(canvas_xy_agent[valid_mask_agent], colors=(255, 200, 100))
    
    canvas_coop.draw_boxes(gt_boxes_corners, colors=(0, 255, 0), texts=[''] * len(gt_boxes_corners))
    
    ax2.axis("off")
    ax2.imshow(canvas_coop.canvas)
    ax2.set_title(f"Ego + Cooperative Vehicles ({record_len} agents)", fontsize=14, pad=10)
    
    ## Subplot 3: BlindMap visualization
    ax3 = plt.subplot(1, 3, 3)
    
    if ego_blind_map is not None:
        # BlindMap coordinate system analysis:
        # - blind_map shape: (grid_size_y, grid_size_x) = (Y_dim, X_dim)
        # - BEV canvas uses image coordinate (Y down), but imshow with origin='upper' also uses Y down
        # - The key issue: when filling BlindMap, Y coordinates may be flipped
        # 
        # BEV Canvas coordinate mapping:
        # - Y increases downward (row index increases downward)
        # - imshow with origin='upper' (default): array[0,0] is top-left
        # 
        # To match BEV canvas (Y down), we use origin='upper'
        
        ego_blind_map_display = ego_blind_map
        
        # Extent: [left, right, bottom, top] in data coordinates
        # With origin='upper': array[0, 0] maps to (X_min, Y_max)
        # So extent should be [X_min, X_max, Y_max, Y_min]
        im = ax3.imshow(ego_blind_map_display, cmap='hot', origin='upper', 
                       extent=[pc_range[0], pc_range[3], pc_range[4], pc_range[1]],
                       aspect='equal')
        ax3.set_xlabel('X (m)', fontsize=12)
        ax3.set_ylabel('Y (m)', fontsize=12)
        ax3.set_title("BlindMap (Occlusion Status)", fontsize=14, pad=10)
        ax3.grid(True, alpha=0.3, color='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Occlusion Intensity', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'No BlindMap Available', 
                ha='center', va='center', fontsize=14)
        ax3.set_title("BlindMap (Not Available)", fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close(fig)


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from: {args.hypes_yaml}")
    hypes = yaml_utils.load_yaml(args.hypes_yaml, None)
    dataset_type = hypes['fusion']['dataset']
    # Get lidar range
    pc_range = hypes['cav_lidar_range']
    print(f"Point cloud range: {pc_range}")
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader with batch_size=1 for visualization
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,  # Use test collate for visualization
        shuffle=False,
        pin_memory=False
    )
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    interval = 20
    total_iterations = num_samples * interval
    print(f"Visualizing {num_samples} samples (1 every {interval} samples) from index {args.idx} to {args.idx + total_iterations}...")
    
    saved_count = 0
    for i, batch_data in enumerate(dataloader):
        if i < args.idx:
            continue
        if i >= args.idx + total_iterations:
            break
        if (i - args.idx) % interval != 0:
            continue
        
        saved_count += 1
        print(f"\nProcessing sample {i} ({saved_count}/{num_samples})...")
        
        # Check if blind_map exists in batch
        if 'blind_map_ego' in batch_data['ego']:
            print(f"  BlindMap shape: {batch_data['ego']['blind_map_ego'].shape}")
            print(f"  BlindMap non-zero: {torch.count_nonzero(batch_data['ego']['blind_map_ego']).item()}")
        else:
            print("  BlindMap not found in batch data")
        
        # Check point cloud
        if 'origin_lidar' in batch_data['ego']:
            print(f"  Origin lidar points: {batch_data['ego']['origin_lidar'].shape}")
        else:
            print("  Origin lidar not found")
        
        # Check ground truth boxes
        if 'object_bbx_center' in batch_data['ego']:
            num_valid = torch.sum(batch_data['ego']['object_bbx_mask'][0] > 0).item()
            print(f"  Valid GT boxes: {num_valid}")
        
        # Check record_len
        if 'record_len' in batch_data['ego']:
            print(f"  Number of agents: {batch_data['ego']['record_len'][0].item()}")
        
        # Visualize
        save_path = os.path.join(args.save_dir, f"{dataset_type}_sample_{i:04d}.png")
        visualize_bev_with_blindmap(batch_data, save_path, pc_range)
    
    print(f"\nVisualization complete! Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()


'''
python opencood/visualization/dataset_vis_occluded.py \
    --hypes_yaml /path/to/config.yaml \
    --save_dir /path/to/output \
    --idx 0 \
    --num_samples 10

实例
python /home/zzh/projects/BlindMap/opencood/visualization/dataset_vis_occluded.py 
    --hypes_yaml /home/zzh/projects/BlindMap/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid_blindmap_noise.yaml 
    --num_samples 10
'''