# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:30:21
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:30:21


import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.pose_utils import generate_noise

class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.use_history = params['use_history']

        # Calculate feature map size the same way as SpVoxelPreprocessor
        self.voxel_size = params["preprocess"]["args"]["voxel_size"]
        self.lidar_range = params["preprocess"]["cav_lidar_range"]
        # Calculate grid size in x,y,z directions
        grid_size = (
            np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])
        ) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # We only need x,y dimensions for blind map
        self.grid_size_y, self.grid_size_x = self.grid_size[1], self.grid_size[0]
        if "use_gaussian" in params["fusion"]["args"].keys():
            self.use_gaussian = self.params["fusion"]["args"]["use_gaussian"]
        else:
            self.use_gaussian = False
        print("use_gaussian:", self.use_gaussian)

        if 'time_delay' in params:          # number of time delay
            self.tau = params['time_delay']
        else:
            self.tau = 0
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 
        self.history_blindmap_dir = os.path.join(
            os.path.dirname(self.root_dir),
            "blindmap_history",
            os.path.basename(self.root_dir)
        )
        self.history_num = params.get("history_num", 3)
        self.history_blindmap_shape = (1, int(self.grid_size_y // 2), int(self.grid_size_x // 2))  # (1, H/2, W/2)
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        
        self.scenario_folders = scenario_folders
        self.reinitialize()


    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                cav_list = sorted(cav_list)  ###为了方便时序BLindmap的离线生成，需要确定ego车辆，所以取消shuffle
                # random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, so here we want to
            make sure they will be in the end of the list as they shouldn't
            be ego vehicle.
            """
            
            """
            make the first cav to be ego modality
            """
            # if getattr(self, "heterogeneous", False):
            #     # print('use heterogeneous data')
            #     scenario_name = scenario_folder.split("/")[-1]
            #     cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]
            # print("cav_list:", cav_list)
            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                
                # this timestamp is not ready
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path, 
                                                timestamp)
                    depth_files = self.find_camera_files(cav_path, 
                                                timestamp, sensor="depth")
                    depth_files = [depth_file.replace("OPV2V", "OPV2V_Hetero") for depth_file in depth_files]

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                    if getattr(self, "heterogeneous", False):
                        scenario_name = scenario_folder.split("/")[-1]

                        cav_modality = self.adaptor.reassign_cav_modality(self.modality_assignment[scenario_name][cav_id] , j)

                        self.scenario_database[i][cav_id][timestamp]['modality_name'] = cav_modality

                        self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                            self.adaptor.switch_lidar_channels(cav_modality, lidar_file)


                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the 
                # scene.
                if j == 0:
                    # print('j==0, cav_id :', cav_id)
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
        # print("len:", self.len_record[-1])
    def get_history_timestamps(self, scenario_database, current_timestamp, history_num=3):
        """
        Get timestamps of history frames.
        
        Parameters
        ----------
        scenario_database : dict
            Database for current scenario
        current_timestamp : str
            Current timestamp
        history_num : int
            Number of history frames to load
            
        Returns
        -------
        history_timestamps : list
            List of history timestamps, sorted from newest to oldest
        """
        # Get all timestamps from first CAV (they should all have same timestamps)
        first_cav_id = list(scenario_database.keys())[0]
        all_timestamps = sorted(list(scenario_database[first_cav_id].keys()))
        all_timestamps.remove('ego')  # Remove 'ego' key
        
        # Find current timestamp index
        current_idx = all_timestamps.index(current_timestamp)
        
        # Get previous timestamps
        history_timestamps = []
        for i in range(1, history_num + 1):
            if current_idx - i >= 0:
                history_timestamps.append(all_timestamps[current_idx - i])
        
        return history_timestamps
    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        ## 假设有3个场景，每个场景的时间戳数分别是: 100, 150, 200
        # self.len_record = [100, 250, 450]  # 累积长度
        '''
                    假设 idx = 80:

            80 < 100 (第一个场景)
            所以 scenario_index = 0
            假设 idx = 180:

            180 > 100 (跳过第一个场景)
            180 < 250 (在第二个场景内)
            所以 scenario_index = 1
            假设 idx = 300:

            300 > 100 (跳过第一个场景)
            300 > 250 (跳过第二个场景)
            300 < 450 (在第三个场景内)
            所以 scenario_index = 2
        '''
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]
        # # Get scenario folder directly using scenario_index
        scenario_folder = self.scenario_folders[scenario_index]
        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        delayed_timestamp_key = self.return_timestamp_key(scenario_database,
                                                            timestamp_index)
        # current frame, wo delay
        curr_timestamp_index = timestamp_index + self.tau + 1 - 1
        # retrieve the corresponding timestamp key
        curr_timestamp_key = self.return_timestamp_key(scenario_database,
                                                    curr_timestamp_index)
        data = OrderedDict()


        # Get ego id - should be the first CAV in sorted list
        cav_list = sorted([x for x in os.listdir(scenario_folder)
                        if os.path.isdir(os.path.join(scenario_folder, x))])
        if int(cav_list[0]) < 0:  # Handle roadside unit case
            cav_list = cav_list[1:] + [cav_list[0]]
            
        ego_id = cav_list[0]  # First CAV is always ego
        
        # Verify ego id
        assert scenario_database[ego_id]['ego'], f"Expected {ego_id} to be ego vehicle"
        # Get history timestamps
        history_timestamps = self.get_history_timestamps(scenario_database, curr_timestamp_key, self.history_num)
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            if self.use_history:
                # Initialize history blindmap list
                data[cav_id]['history_blind_maps'] = []
                # Only load history blind maps for non-ego vehicles
                if not cav_content['ego']:
                    # Load history blind maps from pre-generated files
                    for history_timestamp in history_timestamps:
                        # Construct blind map file path
                        blindmap_folder = os.path.join(self.history_blindmap_dir, f'scenario_{os.path.basename(scenario_folder)}')
                        blindmap_file = os.path.join(blindmap_folder, 
                                                f"ego_{ego_id}_agent_{cav_id}_ts_{history_timestamp}.npy")
                        
                        if os.path.exists(blindmap_file):
                            history_blind_map = np.load(blindmap_file)
                            if history_blind_map.shape != self.history_blindmap_shape:
                                # Create new blind map with target shape
                                new_blind_map = np.zeros(self.history_blindmap_shape)
                                
                                # Get source and target shapes
                                src_h, src_w = history_blind_map[0].shape
                                target_h, target_w = self.history_blindmap_shape[1:]
                                
                                # Calculate dimensions to copy
                                copy_h = min(src_h, target_h)
                                copy_w = min(src_w, target_w)
                                
                                # Calculate starting positions for both source and target
                                src_start_h = (src_h - copy_h) // 2
                                src_start_w = (src_w - copy_w) // 2
                                target_start_h = (target_h - copy_h) // 2
                                target_start_w = (target_w - copy_w) // 2
                                
                                # Copy the overlapping region
                                new_blind_map[0,
                                            target_start_h:target_start_h + copy_h,
                                            target_start_w:target_start_w + copy_w] = \
                                    history_blind_map[0,
                                                    src_start_h:src_start_h + copy_h,
                                                    src_start_w:src_start_w + copy_w]
                                
                                history_blind_map = new_blind_map
                        else:
                            # If file doesn't exist, use zero map
                            print(f"Blind map file not found: {blindmap_file}, using zero map.")
                            history_blind_map = np.zeros(self.history_blindmap_shape)
                            
                        data[cav_id]['history_blind_maps'].append(history_blind_map)
                    # Pad with zeros if needed
                    while len(data[cav_id]['history_blind_maps']) < self.history_num:
                        zero_map = np.zeros(self.history_blindmap_shape)
                        data[cav_id]['history_blind_maps'].append(zero_map)
                else:
                    None
                    # For ego vehicle, we don't load history blind maps
            if cav_content['ego']:
                input_timestamp_key = curr_timestamp_key
            else:
                input_timestamp_key = delayed_timestamp_key
            # load param file: json is faster than yaml
            json_file = cav_content[curr_timestamp_key]['yaml'].replace("yaml", "json")
            occulude_file = cav_content[curr_timestamp_key]['yaml'].replace(".yaml", "_occluded_state.yaml")
            # print('occulude_file:', occulude_file)
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[curr_timestamp_key]['yaml'])
            if os.path.exists(occulude_file):
                data[cav_id]['params_occluded_state'] = load_yaml(occulude_file)
            else:
                None
                # print("occluded state file not found:", occulude_file)
            if cav_content['ego']:
                pass
            else: 
                json_file_lidar_pose = cav_content[input_timestamp_key]['yaml'].replace("yaml", "json")
                if os.path.exists(json_file_lidar_pose):
                    with open(json_file_lidar_pose, "r") as f:
                        data[cav_id]['params']['lidar_pose'] = json.load(f)['lidar_pose']
                else:   
                    data[cav_id]['params']['lidar_pose'] = \
                        load_yaml(cav_content[input_timestamp_key]['yaml'])['lidar_pose']
            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[input_timestamp_key]['cameras'][0].replace("camera0.png", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        if self.load_depth_file:
                            data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]['camera_data'] = \
                        load_camera_data(cav_content[input_timestamp_key]['cameras'])
                if self.load_depth_file:
                    data[cav_id]['depth_data'] = \
                        load_camera_data(cav_content[input_timestamp_key]['depths']) 

            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[input_timestamp_key]['lidar'])

            if getattr(self, "heterogeneous", False):
                data[cav_id]['modality_name'] = cav_content[input_timestamp_key]['modality_name']

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[input_timestamp_key][file_extension]):
                    cav_content[input_timestamp_key][file_extension] = cav_content[input_timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[input_timestamp_key][file_extension] = cav_content[input_timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[input_timestamp_key][file_extension] = cav_content[input_timestamp_key][file_extension].replace("test","additional/test")
                    
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[input_timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[input_timestamp_key][file_extension])

        ##########准备blindmap ground truth生成需要的数据#############
        ego_id = None
        for cav_id, cav_content in data.items():
            if cav_content['ego']:
                ego_id = cav_id
                break
        assert ego_id is not None
        # Generate blind maps
        # print('ego id:', ego_id)
        data = self.generate_blind_map(data, ego_id)

        return data
    def get_vehiclecornors_in_ego_lidar(self, vehicle, ego_pose):
        # Get vehicle pose
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
    def transform_blind_map(self, blind_map, transform_matrix):
        """Transform blind map from one coordinate to another.
        
        Parameters
        ----------
        blind_map : np.ndarray
            Source blind map
        transform_matrix : np.ndarray
            4x4 transformation matrix
            
        Returns
        -------
        np.ndarray
            Transformed blind map
        """
        # Create coordinate meshgrid
        x = np.arange(self.grid_size_x)
        y = np.arange(self.grid_size_y)
        X, Y = np.meshgrid(x, y)
        
        # Convert grid coordinates to real world coordinates
        real_x = X * self.voxel_size[0] + self.lidar_range[0]
        real_y = Y * self.voxel_size[1] + self.lidar_range[1]
        
        # Create homogeneous coordinates
        points = np.stack([real_x, real_y, np.zeros_like(real_x), np.ones_like(real_x)])
        points = points.reshape(4, -1)
        
        # Transform points
        transformed_points = transform_matrix @ points
        
        # Convert back to grid coordinates
        grid_x = ((transformed_points[0] - self.lidar_range[0]) / self.voxel_size[0]).astype(np.int32)
        grid_y = ((transformed_points[1] - self.lidar_range[1]) / self.voxel_size[1]).astype(np.int32)
        
        # Create new blind map
        new_blind_map = np.zeros_like(blind_map)
        valid_mask = (grid_x >= 0) & (grid_x < self.grid_size_x) & \
                    (grid_y >= 0) & (grid_y < self.grid_size_y)
        
        new_blind_map[grid_y[valid_mask], grid_x[valid_mask]] = \
            blind_map[Y.flatten()[valid_mask], X.flatten()[valid_mask]]
        
        return new_blind_map
    def generate_blind_map(self, data, ego_id):
        """Generate blind maps for each agent based on cooperative perception.
    
        For ego vehicle: use occlusion states to mark occluded objects
        For other agents: transform ego's blind map to their coordinates
        
        Parameters
        ----------
        data : dict
            Contains all agent's data
        ego_id : str
            ID of ego vehicle
            
        Returns
        -------
        data : dict
            Updated data with blind maps added
        """
        # # First generate ego's blind map
        # noise_setting = self.params['noise_setting']
        # ego_pose_clean = data[ego_id]["params"]["lidar_pose"]
        # ego_pose = ego_pose_clean + \
        #                         generate_noise(
        #                             noise_setting['args']['pos_std'],
        #                             noise_setting['args']['rot_std'],
        #                             noise_setting['args']['pos_mean'],
        #                             noise_setting['args']['rot_mean']
        #                         )
        ego_pose = data[ego_id]["params"]["lidar_pose"]
        world_to_ego = np.linalg.inv(x_to_world(ego_pose))
        
        # Initialize ego's blind map
        ego_blind_map = np.zeros((self.grid_size_y, self.grid_size_x))
        
        # Collect all vehicles from all agents
        all_others_vehicles = {}
        for cav_id, cav_data in data.items():
            if cav_id != ego_id:
                vehicles = cav_data["params"]["vehicles"]
                # print('vehicles:', vehicles)
                all_others_vehicles.update(vehicles)
        
        # Process ego's blind map using occlusion states
        if 'params_occluded_state' in data[ego_id]:
            ego_vehicles_occluded_states = data[ego_id]['params_occluded_state']['vehicles']
            
            for vehicle_id, vehicle in ego_vehicles_occluded_states.items():
                #针对ego自己的目标，根据occluded_state在blindmap中填充
                if ego_vehicles_occluded_states[vehicle_id]["occluded_state"] > 0:
                    vehicle_vorners_in_ego = self.get_vehiclecornors_in_ego_lidar(vehicle, ego_pose)
                    # if self.train:
                    #     vehicle_corners_in_ego = self._add_bbox_noise(vehicle_corners_in_ego)
                    ego_blind_map = self.fill_box_in_blindmap(ego_blind_map, vehicle_vorners_in_ego)
            for other_vehicle_id, other_vehicle in all_others_vehicles.items():
                # 先去除包含在ego车辆遮挡状态中的车辆
                if other_vehicle_id in ego_vehicles_occluded_states:
                    continue
                else:
                    other_vehicle_corners_in_ego = self.get_vehiclecornors_in_ego_lidar(other_vehicle, ego_pose)
                    # if self.train:
                    #     other_vehicle_corners_in_ego = self._add_bbox_noise(other_vehicle_corners_in_ego)
                    ego_blind_map = self.fill_box_in_blindmap(ego_blind_map, other_vehicle_corners_in_ego)
        else:
            print("No occluded state data available for ego vehicle. maybe no vehicles around")
            for other_vehicle_id, other_vehicle in all_others_vehicles.items():
                other_vehicle_corners_in_ego = self.get_vehiclecornors_in_ego_lidar(other_vehicle, ego_pose)
                ego_blind_map = self.fill_box_in_blindmap(ego_blind_map, other_vehicle_corners_in_ego)
        # Store ego's blind map
        # print('ego blindmap shape:', ego_blind_map.shape)
        # cav blindmap shape: (256, 512)
        # 策略5: 对整个blind map进行后处理
        # if self.train:
        #     ego_blind_map = self._apply_blind_map_augmentation(ego_blind_map)
        data[ego_id]["blind_map"] = ego_blind_map
        
        # Transform ego's blind map to other agents' coordinates
        for cav_id in data.keys():
            if cav_id != ego_id:
                cav_pose = data[cav_id]["params"]["lidar_pose"]
                # Calculate transformation from ego to current agent
                world_to_cav = np.linalg.inv(x_to_world(cav_pose))
                ego_to_cav = world_to_cav @ np.linalg.inv(world_to_ego)
                
                # Transform ego's blind map to current agent's coordinate
                cav_blind_map = self.transform_blind_map(ego_blind_map, ego_to_cav)
                # print('cav blindmap shape:', cav_blind_map.shape)
                #cav blindmap shape: (256, 512)
                data[cav_id]["blind_map"] = cav_blind_map
        
        return data
    def _add_bbox_noise(self, corners, noise_scale=0.2):
        """为边界框添加随机噪声"""
        noise = np.random.normal(0, noise_scale, corners.shape)
        return corners + noise
    def _apply_blind_map_augmentation(self, blind_map):
        """对blind map进行数据增强"""
        augmented_map = blind_map.copy()
        
        # 策略5a: 随机擦除小块区域
        if np.random.random() < 0.1:  # 10%概率
            h, w = blind_map.shape
            erase_h = np.random.randint(5, 15)
            erase_w = np.random.randint(5, 15)
            start_h = np.random.randint(0, max(1, h - erase_h))
            start_w = np.random.randint(0, max(1, w - erase_w))
            augmented_map[start_h:start_h+erase_h, start_w:start_w+erase_w] = 0
        
        # 策略5b: 添加随机噪声点
        if np.random.random() < 0.1:  # 10%概率
            noise_points = np.random.randint(0, 2, size=blind_map.shape) * 0.1
            augmented_map = np.clip(augmented_map + noise_points, 0, 1)
        
        # 策略5c: 轻微的形态学操作
        if np.random.random() < 0.1:  # 10%概率
            from scipy import ndimage
            # 随机选择腐蚀或膨胀
            if np.random.random() < 0.5:
                augmented_map = ndimage.binary_erosion(augmented_map > 0.5).astype(float)
            else:
                augmented_map = ndimage.binary_dilation(augmented_map > 0.5).astype(float)
        
        return augmented_map

    def world_to_feature_coords(self, x, y):
        """Convert world coordinates to feature map coordinates"""
        # Convert from world coordinates to feature map coordinates
        # lidar_range: [-100.8, -40, -3, 100.8, 40, 1] (xmin,ymin,zmin,xmax,ymax,zmax)
        x_range = self.lidar_range[3] - self.lidar_range[0]
        y_range = self.lidar_range[4] - self.lidar_range[1]

        # Normalize to [0, 1] then scale to grid size
        x_feature = int((x - self.lidar_range[0]) / x_range * self.grid_size_x)
        y_feature = int((y - self.lidar_range[1]) / y_range * self.grid_size_y)

        return x_feature, y_feature
    def fill_box_in_blindmap(self, blind_map, points):
        """Fill polygon area in blind map using world_8_points"""
        # points: array of shape (8,3) containing x,y,z coordinates

        feature_points = []
        # Only need x,y coordinates for BEV
        for point in points[:4]:  # Only need bottom 4 points for BEV
            x_feature, y_feature = self.world_to_feature_coords(point[0], point[1])
            feature_points.append([x_feature, y_feature])

        feature_points = np.array(feature_points)

        # Use cv2.fillPoly to fill the area
        if not self.use_gaussian:
            cv2.fillPoly(blind_map, [feature_points.astype(np.int32)], 1)
        # OPTIONAL: 2d gaussian box_2d_center radius
        else:
            # Calculate box center in feature map coordinates
            box_center = np.mean(feature_points, axis=0).astype(int)
            x_center, y_center = box_center

            # Calculate box size to determine sigma
            width = max(np.max(feature_points[:, 0]) - np.min(feature_points[:, 0]), 1)
            height = max(np.max(feature_points[:, 1]) - np.min(feature_points[:, 1]), 1)

            # Set sigma proportional to box size
            sigma_x = width / self.sigma  # 3-sigma rule
            sigma_y = height / self.sigma

            # Create a Gaussian kernel
            x_grid = np.arange(0, self.grid_size_x)
            y_grid = np.arange(0, self.grid_size_y)
            x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

            # Calculate Gaussian distribution
            gaussian = np.exp(
                -(
                    (x_mesh - x_center) ** 2 / (2 * sigma_x**2)
                    + (y_mesh - y_center) ** 2 / (2 * sigma_y**2)
                )
            )

            # Normalize to [0, 1] and apply it to regions outside the box
            ma_gaussian, mi_gaussian = np.max(gaussian), np.min(gaussian)
            if ma_gaussian != mi_gaussian:
                gaussian = (gaussian - mi_gaussian) / (ma_gaussian - mi_gaussian)
                blind_map = np.maximum(blind_map, gaussian)
        return blind_map


    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """

        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            if 'occluded_state' in timestamp:
                continue
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # # get all timestamp keys
        # timestamp_keys = list(scenario_database.items())[0][1]
        # # retrieve the correct index
        # timestamp_key = list(timestamp_keys.items())[timestamp_index][0]
         # 只保留时间戳 key，排除 'ego'
        timestamp_keys = [k for k in list(scenario_database.items())[0][1].keys() if k != 'ego']
        # 防止越界
        if timestamp_index >= len(timestamp_keys):
            timestamp_index = len(timestamp_keys) - 1
        timestamp_key = timestamp_keys[timestamp_index]
        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic