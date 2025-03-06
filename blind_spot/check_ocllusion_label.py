
import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from torch.utils.data import DataLoader

import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic

from torchsummary import summary
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", default='/home/zzh/projects/Where2comm/opencood/hypes_yaml/dair-v2x/zzh_dair_where2comm_max_resnet.yaml',type=str,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt
def check_dataset(hypes):

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_train_dataset.some_test()
    # print(opencood_train_dataset[0])
    train_loader = DataLoader(opencood_train_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    for i, batch_data in enumerate(train_loader):
        
        if i>0:
            break
        print('voxel_features shape:', (batch_data['ego']['processed_lidar']['voxel_features'].shape))
        print('voxel_coords shape:', (batch_data['ego']['processed_lidar']['voxel_coords'].shape))
        print('voxel_num_points shape:', (batch_data['ego']['processed_lidar']['voxel_num_points'].shape))
        print('record_len shape:', (batch_data['ego']['record_len'].shape))
        print('pairwise_t_matrix shape:', (batch_data['ego']['pairwise_t_matrix'].shape))


        print(len(batch_data['ego']['object_bbx_center_single_v']))
        print(len(batch_data['ego']['object_bbx_center_single_i']))
        # dict_keys(['object_bbx_center', 'object_bbx_mask', 'object_ids', 'label_dict', 
        # 'object_bbx_center_single_v', 'object_bbx_mask_single_v', 'object_ids_single_v', 'label_dict_single_v', 
        # 'object_bbx_center_single_i', 'object_bbx_mask_single_i', 'object_ids_single_i', 'label_dict_single_i', 
        # 'processed_lidar', 'record_len', 'pairwise_t_matrix', 'lidar_pose_clean', 'lidar_pose'])
    return train_loader


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    train_loader = check_dataset(hypes)
    
    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    # print(model)
    for i, batch_data in enumerate(train_loader):
        batch_data = train_utils.to_device(batch_data, device)
        if i>0:
            break
        output_dict = model(batch_data['ego'])
        
    
if __name__ == '__main__':
    main()
