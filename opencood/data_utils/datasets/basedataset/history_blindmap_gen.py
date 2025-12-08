# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:30:07
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:30:07


import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.data_utils.datasets.blindmap_intermediate_heter_v2xset_fusion_dataset import getBlindmapintermediateheterv2xsetFusionDataset
class BlindMapGenerator:
    def __init__(self, hypes_yaml, model_path, data_split):
        """
        Initialize generator with trained model

        Parameters
        ----------
        hypes_yaml : str
            Path to config file
        model_path : str
            Path to model weights
        data_split : str
            Which data split to use ('train', 'validate', or 'test')
        """
        self.hypes = load_yaml(hypes_yaml)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hypes['validate_dir'] = self.hypes['validate_dir'].replace('validate', data_split)
        self.hypes['test_dir'] = self.hypes['test_dir'].replace('test', data_split)
        self.hypes['root_dir'] = self.hypes['root_dir'].replace('train', data_split)
        # Create dataset and dataloader following train_zzh_test_v2xset.py
        opencood_dataset = OPV2VBaseDataset
        BasedatasetClass = getBlindmapintermediateheterv2xsetFusionDataset(opencood_dataset)
        self.dataset = BasedatasetClass(self.hypes, visualize=False, train=False)
        
        self.dataloader = DataLoader(self.dataset,
                                   batch_size=1,
                                   num_workers=4,
                                   collate_fn=self.dataset.collate_batch_test,
                                   shuffle=False,
                                   pin_memory=False,
                                   drop_last=False)
        
        # Load trained model
        self.model = train_utils.create_model(self.hypes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
    def get_timestamp_list(self, scenario_folder, cav_id):
        """
        Get list of timestamps from yaml files in CAV folder
        
        Parameters
        ----------
        scenario_folder : str
            Path to scenario folder
        cav_id : str
            CAV ID to get timestamps from
            
        Returns
        -------
        list
            List of timestamps (sorted)
        """
        cav_path = os.path.join(scenario_folder, cav_id)
        yaml_files = [x for x in os.listdir(cav_path) 
                    if x.endswith('.yaml') and 'additional' not in x]
        # Extract timestamps from yaml files (remove .yaml extension)
        timestamps = [x.replace('.yaml', '') for x in yaml_files]
        # Sort timestamps
        timestamps.sort()
        return timestamps
    def generate_blindmaps(self, save_path):
        """
        Generate and save blindmaps for each scenario
        """
        os.makedirs(save_path, exist_ok=True)
    
        for batch in self.dataloader:
            if batch is None:
                continue
                
            # Get meta information for each sample in batch
            meta_info_list = batch['ego']['meta_info_list']
            cav_id_lists = batch['ego']['cav_id_lists']
            
            # Process each sample in batch
            for sample_idx in range(len(meta_info_list)):
                meta_info = meta_info_list[sample_idx]
                cav_id_list = cav_id_lists[sample_idx]
                
                scenario_folder = meta_info['scenario_folder']
                timestamp = meta_info['timestamp']
                ego_id = meta_info['ego_id']
                
                # Create scenario directory
                scenario_id = os.path.basename(scenario_folder)
                scenario_save_path = os.path.join(save_path, f"scenario_{scenario_id}")
                os.makedirs(scenario_save_path, exist_ok=True)
                
                # Convert batch to device
                
                batch_for_model = train_utils.to_device(batch, self.device)
                
                # Generate blindmap
                with torch.no_grad():
                    output_dict = self.model(batch_for_model['ego'])
                    blindmaps = output_dict['pred_blind_maps']
                    
                    if blindmaps is not None:
                        # Save blindmaps for each agent
                        for agent_idx, agent_id in enumerate(cav_id_list):
                            if agent_id != ego_id:  # Skip ego vehicle
                                blindmap = blindmaps[agent_idx].cpu().numpy()
                                save_name = f"ego_{ego_id}_agent_{agent_id}_ts_{timestamp}.npy"
                                save_file = os.path.join(scenario_save_path, save_name)
                                np.save(save_file, blindmap)
                                
                print(f"Processed scenario {scenario_id}, timestamp {timestamp}")

def main():
    
    hypes_yaml = "/home/zzh/projects/HEAL/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid_history_blindmap_gen.yaml"
    model_path = "/home/zzh/projects/HEAL/opencood/logs/HeterBaseline_opv2v_lidar_pyramid_blindmap_2025_06_09_20_22_05_thre_0.01_add_noise/net_epoch_bestval_at21.pth"
    base_save_path = "//home/zzh/projects/HEAL/dataset/OPV2V/blindmap_history"
    for split in ['train', 'validate', 'test']:
        print(f"\nProcessing {split} set...")
        
        # Create generator for current split
        generator = BlindMapGenerator(hypes_yaml, model_path, split)
        
        # Set save path for current split
        save_path = os.path.join(base_save_path, split)
        os.makedirs(save_path, exist_ok=True)
        
        # Generate blindmaps
        generator.generate_blindmaps(save_path)
        print(f"Finished processing {split} set")

if __name__ == "__main__":
    main()