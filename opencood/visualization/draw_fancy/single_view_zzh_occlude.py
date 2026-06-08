
from email.mime import base
from tkinter import Y
import cv2

# from black import left_hand_split
from opencood.utils.transformation_utils import x_to_world, x1_to_x2
from opencood.utils.box_utils import create_bbx, mask_boxes_outside_range_numpy, corner_to_center
from torch.utils.data import Subset
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
import numpy as np
import os
import copy
from pyquaternion import Quaternion

v2x = False
if v2x:
    from opencood.visualization.draw_fancy.draw_fancy_datasetv2x import SimpleDataset
else:
    from opencood.visualization.draw_fancy.draw_fancy_dataset import SimpleDataset
from opencood.visualization.draw_fancy.draw_fancy_dataset_zzh import ZZHSimpleDataset
COLOR = ['red','springgreen','dodgerblue', 'darkviolet']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
classes = ['agent1', 'agent2', 'agent3', 'agent4']


def generate_object_center_v2x(cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (1, 8, 3).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE

        order = 'hwl'
        max_num = 200
        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        print("gt_boxes", gt_boxes)
        object_ids = cav_contents[0]['params']['object_ids']
        
        object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        lidar_range = (-64,-64,-3,64,64,2)
        
        gt_boxes = object_dict['gt_boxes']
        object_ids = object_dict['object_ids']
        for i, object_content in enumerate(gt_boxes):
            x,y,z,dx,dy,dz,w,a,b,c = object_content

            q = Quaternion([w,a,b,c])
            T_world_object = q.transformation_matrix
            T_world_object[:3,3] = object_content[:3]

            T_world_lidar = x_to_world(reference_lidar_pose)

            object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object


            # shape (3, 8)
            # hopefully this is correct? 
            x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
            y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
            z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

            bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate
            bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
            bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)

            bbox_corner = copy.deepcopy(bbx_lidar)

            bbx_lidar = corner_to_center(bbx_lidar, order=order)
            bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                    lidar_range,
                                                    order)


            if bbx_lidar.shape[0] > 0:
                output_dict.update({object_ids[i]: bbox_corner})


        object_np = np.zeros((max_num, 8, 3))
        mask = np.zeros(max_num)
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        # should not appear repeated items
        object_np = object_np[:len(object_ids)]

        return object_np, object_ids

def generate_object_center(cav_contents,
                            reference_lidar_pose):
    """
    Retrieve all objects in a format of (n, 7), where 7 represents
    x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

    Parameters
    ----------
    cav_contents : list
        List of dictionary, save all cavs' information.

    reference_lidar_pose : list
        The final target lidar pose with length 6.

    Returns
    -------
    object_np : np.ndarray
        Shape is (n, 8, 3). n is number of xxx

    object_ids : list
        Length is number of bbx in current sample.
    """
    
    order = 'hwl'
    max_num = 200


    tmp_object_dict = {}
    for cav_content in cav_contents:
        tmp_object_dict.update(cav_content['params']['vehicles'])
    # print("tmp_object_dict", tmp_object_dict.keys())
    output_dict = {}
    filter_range = [-140, -60, -3, 140, 60, 2]

    for object_id, object_content in tmp_object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, reference_lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)

        bbox_corner = copy.deepcopy(bbx_lidar)

        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                   filter_range,
                                                   order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbox_corner})

    object_np = np.zeros((max_num, 8, 3))
    mask = np.zeros(max_num)
    object_ids = []

    for i, (object_id, object_bbx) in enumerate(output_dict.items()):
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        object_ids.append(object_id)

    unique_indices = \
                [object_ids.index(x) for x in set(object_ids)]
    near_indices = [idx for idx in unique_indices if (object_np[idx][0][0]**2 + object_np[idx][0][1]**2) < 45**2]
    print(len(unique_indices), len(near_indices))
    object_np = object_np[near_indices]

    return object_np, near_indices

def get_vehiclecornors_in_ego_lidar(vehicle, ego_pose):
    """Get vehicle corners in ego lidar coordinate system"""
    loc = vehicle['location']
    center = vehicle.get('center', [0, 0, 0])
    angles = vehicle['angle']
    
    # Construct vehicle pose
    object_pose = [
        loc[0] + center[0],  # x
        loc[1] + center[1],  # y
        loc[2] + center[2],  # z
        angles[0],           # roll
        angles[1],           # pitch
        angles[2]            # yaw
    ]
    
    # Get transformation matrix from object to ego
    object2ego = x1_to_x2(object_pose, ego_pose)

    # Create corners in object's local coordinate
    extent = vehicle['extent']  # [l/2, w/2, h/2]
    l, w, h = extent[0]*2, extent[1]*2, extent[2]*2
    corners_local = np.array([
        [ l/2, -w/2, -h/2, 1],  # front left bottom
        [ l/2,  w/2, -h/2, 1],  # front right bottom
        [-l/2,  w/2, -h/2, 1],  # rear right bottom
        [-l/2, -w/2, -h/2, 1],  # rear left bottom
        [ l/2, -w/2,  h/2, 1],  # front left top
        [ l/2,  w/2,  h/2, 1],  # front right top
        [-l/2,  w/2,  h/2, 1],  # rear right top
        [-l/2, -w/2,  h/2, 1]   # rear left top
    ])

    # Transform corners to ego coordinate
    ego_corners = (object2ego @ corners_local.T).T[:, :3]
    return ego_corners

def world_to_feature_coords(x, y, lidar_range, grid_size_x, grid_size_y):
    """Convert world coordinates to feature map coordinates"""
    x_range = lidar_range[3] - lidar_range[0]
    y_range = lidar_range[4] - lidar_range[1]

    # Normalize to [0, 1] then scale to grid size
    x_feature = int((x - lidar_range[0]) / x_range * grid_size_x)
    y_feature = int((y - lidar_range[1]) / y_range * grid_size_y)

    return x_feature, y_feature

def fill_box_in_blindmap(blind_map, points, lidar_range, grid_size_x, grid_size_y):
    """Fill polygon area in blind map using world_8_points"""
    feature_points = []
    # Only need x,y coordinates for BEV
    for point in points[:4]:  # Only need bottom 4 points for BEV
        x_feature, y_feature = world_to_feature_coords(point[0], point[1], lidar_range, grid_size_x, grid_size_y)
        feature_points.append([x_feature, y_feature])

    feature_points = np.array(feature_points)

    # Use cv2.fillPoly to fill the area
    cv2.fillPoly(blind_map, [feature_points.astype(np.int32)], 1)
    return blind_map

def generate_blind_map(base_data_dict, ego_id, voxel_size=[0.4, 0.4, 4], lidar_range=[-102.4, -102.4, -3, 102.4, 102.4, 1]):
    """Generate blind map for ego vehicle"""
    # Calculate grid size
    grid_size = (
        np.array(lidar_range[3:6]) - np.array(lidar_range[0:3])
    ) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    grid_size_y, grid_size_x = grid_size[1], grid_size[0]
    
    # Initialize blind map
    ego_blind_map = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
    
    # Get ego pose
    ego_pose = base_data_dict[ego_id]['params']['lidar_pose']
    
    # Collect all other vehicles
    all_others_vehicles = {}
    for cav_id, cav_data in base_data_dict.items():
        if cav_id != ego_id:
            vehicles = cav_data["params"]["vehicles"]
            all_others_vehicles.update(vehicles)
    
    # Process ego's blind map using occlusion states
    if 'params_occluded_state' in base_data_dict[ego_id]:
        ego_vehicles_occluded_states = base_data_dict[ego_id]['params_occluded_state']['vehicles']
        
        for vehicle_id, vehicle in ego_vehicles_occluded_states.items():
            # Fill occluded vehicles in blindmap
            if ego_vehicles_occluded_states[vehicle_id]["occluded_state"] > 0:
                vehicle_corners_in_ego = get_vehiclecornors_in_ego_lidar(vehicle, ego_pose)
                ego_blind_map = fill_box_in_blindmap(ego_blind_map, vehicle_corners_in_ego, 
                                                     lidar_range, grid_size_x, grid_size_y)
        
        # Add other vehicles not in ego's field of view
        for other_vehicle_id, other_vehicle in all_others_vehicles.items():
            if other_vehicle_id not in ego_vehicles_occluded_states:
                other_vehicle_corners_in_ego = get_vehiclecornors_in_ego_lidar(other_vehicle, ego_pose)
                ego_blind_map = fill_box_in_blindmap(ego_blind_map, other_vehicle_corners_in_ego,
                                                     lidar_range, grid_size_x, grid_size_y)
    else:
        print("No occluded state data available for ego vehicle.")
        for other_vehicle_id, other_vehicle in all_others_vehicles.items():
            other_vehicle_corners_in_ego = get_vehiclecornors_in_ego_lidar(other_vehicle, ego_pose)
            ego_blind_map = fill_box_in_blindmap(ego_blind_map, other_vehicle_corners_in_ego,
                                                 lidar_range, grid_size_x, grid_size_y)
    
    return ego_blind_map

def main():
    ## basic setting
    path = '/home/zzh/projects/HEAL/dataset/V2XSET/train/2021_08_23_13_10_47'
    agent = '127694954'
    time = '000070'
    dataset = ZZHSimpleDataset()
    

    

    ## matplotlib setting
    plt.figure()
    plt.style.use('dark_background')

    ## box setting
    # ego coord
    dx = 4.9
    dy = 2
    dz = 1.5
    x_corners = dx / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])  # (8,)
    y_corners = dy / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = dz / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    box_corners = np.stack((x_corners, y_corners, z_corners), axis=-1) # (8, 3)
    # box_corners = np.pad(box_corners,((0,0),(0,1)), constant_values=1) # (8, 4)
    box_corners = box_corners[np.newaxis,...]
    if v2x:
        box_corners[:,:,0] -= 2.2


    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset.get_agent_timestamp(path, agent, time)
        cav_ids = list(base_data_dict.keys())
        cav_invert_dict = dict() # cav_id -> 0/1/2
        for (idx, cav_id) in enumerate(cav_ids):
            cav_invert_dict[cav_id] = idx
        recs = []
        for i in range(0,len(cav_ids)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))
        #print(base_data_dict.keys())
        # ['3242', '3251', '3260', '3269']
        
        # retrieve all bbox, under world coordinate
        for cav_id, cav_content in base_data_dict.items():
            lidar_np_ego_agg = np.zeros((0, 4))
            cav_box_agg = dict()
            cav_lidar_agg = dict()
            ego_pose = cav_content['params']['lidar_pose']
            ego_id = cav_id

            if v2x:
                cav_contents = [base_data_dict[1]]
            else:
                cav_contents = [cav_content]

            if v2x:
                object_np, object_ids = generate_object_center_v2x(cav_contents, ego_pose)
            else:
                object_np, object_ids = generate_object_center(cav_contents, ego_pose)

            if (not v2x) and (not cav_id in object_ids):
                object_np = np.concatenate((object_np, box_corners), axis=0)
                object_ids.append(cav_id)

            lidar_np_ego_agg = cav_content['lidar_np']

            # Generate BlindMap for the current ego vehicle
            print(f"Generating BlindMap for agent {cav_id}...")
            voxel_size = [0.4, 0.4, 4]
            lidar_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]  # Match OPV2V dataset range
            ego_blind_map = generate_blind_map(base_data_dict, cav_id, voxel_size, lidar_range)
            print(f"BlindMap shape: {ego_blind_map.shape}, Non-zero pixels: {np.count_nonzero(ego_blind_map)}")

            ## Create figure with 2 subplots: BEV point cloud + bbox (top), and blindmap (bottom)
            fig = plt.figure(figsize=(10, 16))
            
            ## Top subplot: BEV point cloud and bboxes
            ax1 = plt.subplot(2, 1, 1)
            
            ## setting canvas and pc_range for BEV
            pc_range = lidar_range  # [-140, -60, -3, 140, 60, 2]
            
            if v2x:
                left_hand = False
            else:
                left_hand = True

            # Use BEV canvas like simple_vis.py
            canvas = canvas_bev.Canvas_BEV_heading_right(
                canvas_shape=(int((pc_range[4]-pc_range[1])*10), int((pc_range[3]-pc_range[0])*10)),
                canvas_x_range=(pc_range[0], pc_range[3]), 
                canvas_y_range=(pc_range[1], pc_range[4]),
                left_hand=left_hand
            )

            canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_ego_agg)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            
            # draw bbox for each cav
            if cav_id == 3:
                object_np = np.concatenate((object_np, box_corners), axis=0)
            canvas.draw_boxes(object_np, colors=COLOR_RGB[cav_invert_dict[cav_id]])

            ax1.axis("off")
            ax1.imshow(canvas.canvas)
            ax1.set_title(f"BEV Point Cloud & BBoxes - Agent {cav_id}", color='black', fontsize=14, pad=10)
            
            ## Bottom subplot: BlindMap visualization
            ax2 = plt.subplot(2, 1, 2)
            # Transpose and flip BlindMap to match BEV canvas orientation
            # BEV canvas: X goes right (columns), Y goes up (rows)
            # BlindMap: needs to match this orientation
            ego_blind_map_display = np.flipud(ego_blind_map.T)  # Transpose and flip vertically
            
            # Display BlindMap with a color map (occluded regions will be bright)
            im = ax2.imshow(ego_blind_map_display, cmap='hot', origin='lower', 
                           extent=[lidar_range[1], lidar_range[4], lidar_range[0], lidar_range[3]],
                           aspect='equal')
            ax2.set_xlabel('Y (m)', fontsize=12)
            ax2.set_ylabel('X (m)', fontsize=12)
            ax2.set_title(f"BlindMap (Occlusion Status) - Agent {cav_id}", fontsize=14, pad=10)
            ax2.grid(True, alpha=0.3)
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Occlusion Intensity', fontsize=11)

            plt.tight_layout()
            path1 = os.path.join('blindmap_visualization', os.path.basename(path))
            if v2x:
                # save_path = f"./result_v2x/single_view_{classes[cav_invert_dict[cav_id]]}"
                save_path = os.path.join(path1, f'{cav_id}_{time}single_view_with_blindmap')
            else:
                # save_path = f"./result/single_view_{classes[cav_invert_dict[cav_id]]}"
                save_path = os.path.join(path1, f'{cav_id}_{time}single_view_with_blindmap')

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            plt.savefig(f"{save_path}/{idx:02d}.png", transparent=False, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}/{idx:02d}.png")
            plt.close(fig)
        break

if __name__ == "__main__":

    main()