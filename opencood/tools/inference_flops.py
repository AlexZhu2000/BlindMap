# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:28:32
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:28:32


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")

    parser.add_argument("--comm_volume_MB", type=float, default=None)
    parser.add_argument( "--comm_thre", type=float, default=None, help="Communication confidence threshold",)
    opt = parser.parse_args()
    return opt

class WrappedModel(torch.nn.Module):
    def __init__(self, model, example_input):
        super().__init__()
        self.model = model
        self.example_input = example_input

    def forward(self, *args):
        # 直接使用真实的 dict
        return self.model(self.example_input)
def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    # # 如果设置了通信阈值，则更新配置文件中的通信阈值
    # if opt.comm_thre is not None:
    #     hypes["model"]["args"]["fusion_backbone"]["communication"]["thre"] = opt.comm_thre
    #     hypes["model"]["args"]["fusion_backbone"]["communication"]['use_threshold'] = True
    # # 如果设置了通信体积，则更新配置文件中的通信体积
    # if opt.comm_volume_MB is not None:
    #     hypes["model"]["args"]["fusion_backbone"]["communication"]["comm_volume_MB"] = (opt.comm_volume_MB)
    #     hypes["model"]["args"]["fusion_backbone"]["communication"]["use_threshold"] = False
    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    

    infer_info = opt.fusion_method + opt.note

    total_comm_rates = []
    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            wrapped_model = WrappedModel(model, batch_data['ego'])
            import thop
            flops, params = thop.profile(wrapped_model,inputs=(batch_data['ego'],))
            break
        torch.cuda.empty_cache()
    # Create text file to save results
    txt_path = os.path.join(saved_path, f'{resume_epoch}_flops_params.txt')
    with open(txt_path, 'w') as f:
        # Write FLOPs and parameters
        f.write(f'FLOPs: {flops/1e9:.2f}G\n')  # Convert to billions
        f.write(f'Parameters: {params/1e6:.2f}M\n\n')  # Convert to millions
        
        # Write model structure
        f.write('Model Structure:\n')
        f.write('-' * 50 + '\n')
        f.write(str(model))
        
    print(f"FLOPs and parameters saved to {txt_path}")
if __name__ == '__main__':
    main()
