# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

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

import opencood.tools.inference_runtime_config as runtime_config

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
    parser.add_argument('--noise', default="0,0,0,0", type=str, help="pose error")
    parser.add_argument("--time_delay", type=int, default=0, help="Time delay for the communication")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    # 如果设置了通信阈值，则更新配置文件中的通信阈值
    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_backbone"]["communication"]["thre"] = opt.comm_thre
        hypes["model"]["args"]["fusion_backbone"]["communication"]['use_threshold'] = True
    # 如果设置了通信体积，则更新配置文件中的通信体积
    if opt.comm_volume_MB is not None:
        hypes["model"]["args"]["fusion_backbone"]["communication"]["comm_volume_MB"] = (opt.comm_volume_MB)
        hypes["model"]["args"]["fusion_backbone"]["communication"]["use_threshold"] = False
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

    hypes['time_delay'] = opt.time_delay
    delay_time_ms = hypes['time_delay'] * 100
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
    
    runtime_config.saved_path = saved_path
    
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    noise_setting = OrderedDict()
    noise_opt = opt.noise.split(',')
    assert len(noise_opt) == 4
    pos_std, rot_std, pos_mean, rot_mean = (float(x) for x in noise_opt)
    noise_args = {'pos_std': 0,
                    'rot_std': 0,
                    'pos_mean': 0,
                    'rot_mean': 0}
    noise_setting['add_noise'] = False if opt.noise == '0,0,0,0' else True
    noise_args['pos_std'] = pos_std
    noise_args['rot_std'] = rot_std
    noise_args['pos_mean'] = pos_mean
    noise_args['rot_mean'] = rot_mean
    noise_setting['args'] = noise_args
    hypes.update({'noise_setting': noise_setting})
    print(hypes['noise_setting'])
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
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note
    model_times = []
    total_comm_rates = []
    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                start = time.time()
                infer_result, comm_rates = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                model_time = time.time() - start
                model_times.append(model_time)
                total_comm_rates.append(comm_rates)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i< 50 or i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                vis_save_path_blindmap = os.path.join(vis_save_path_root, 'bev_%05d_blindmap.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                # simple_vis.visualize_blindmap(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path_blindmap,
                #                     method='bev',
                #                     left_hand=left_hand)
        torch.cuda.empty_cache()
    if len(total_comm_rates) > 0:
        comm_rates = sum(total_comm_rates) / len(total_comm_rates)
        if hasattr(comm_rates, "item"):
            comm_rates = comm_rates.item()
    else:
        comm_rates = 0
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    model_time_av = sum(model_times)/len(model_times)
    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}'.format(resume_epoch, ap30, ap50, ap70, comm_rates)
            end = (
            " | comm_thre: {:.04f}".format(opt.comm_thre)
            if opt.comm_thre is not None and opt.comm_volume_MB is None
            else " | comm_volume_MB: {}".format(opt.comm_volume_MB))
            if opt.comm_thre is not None or opt.comm_volume_MB is not None:
                msg = msg + end
            _range = " |# " + opt.range + " | # "
            # Convert noise settings to string
            if hypes['noise_setting']['add_noise']:
                noise_args = hypes['noise_setting']['args']
                noise_str = '_'.join([f"{k}={v}" for k, v in noise_args.items()])
            else:
                noise_str = 'no_noise'
            time_delay = 'delay' + str(delay_time_ms) + 'ms' if delay_time_ms > 0 else 'no_delay'
            time_av = '' if model_time_av == 0 else ' | time_av: {:.04f}'.format(model_time_av)
            msg += (_range + noise_str +time_delay + time_av + '\n') 
            f.write(msg)
            print(msg)
if __name__ == '__main__':
    main()
