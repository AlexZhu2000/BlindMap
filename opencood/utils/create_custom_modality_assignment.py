#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建自定义模态分配文件
确保ego车辆为lidar，所有协同车辆为camera
"""

import json
import os
import sys
from collections import OrderedDict


def create_ego_lidar_others_camera_assignment(
    root_dir="/home/zzh/projects/HEAL/dataset/OPV2V",
    output_path="opencood/logs/heter_modality_assign/opv2v_ego_lidar_others_camera.json"
):
    """
    为每个场景创建模态分配：
    - 第一个CAV (ego): m1 (lidar)
    - 所有其他CAV: m2 (camera)
    
    Args:
        root_dir: 数据集根目录
        output_path: 输出JSON文件路径
    """
    splits = ['train', 'test', 'validate']
    scenario_cav_modality_dict = OrderedDict()
    
    total_scenarios = 0
    total_agents = 0
    ego_count = 0
    camera_count = 0

    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            print(f"警告: 路径不存在 {split_path}，跳过")
            continue
            
        scenario_folders = sorted([
            os.path.join(split_path, x)
            for x in os.listdir(split_path) 
            if os.path.isdir(os.path.join(split_path, x))
        ])

        print(f"\n处理 {split} 集: {len(scenario_folders)} 个场景")

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([
                x for x in os.listdir(scenario_folder) 
                if os.path.isdir(os.path.join(scenario_folder, x))
            ])

            total_scenarios += 1
            total_agents += len(cav_list)

            for j, cav_id in enumerate(cav_list):
                if j == 0:
                    # 第一个CAV设为m1 (将成为ego，使用lidar)
                    scenario_cav_modality_dict[scenario_name][cav_id] = 'm1'
                    ego_count += 1
                else:
                    # 其他所有CAV设为m2 (协同车，使用camera)
                    scenario_cav_modality_dict[scenario_name][cav_id] = 'm2'
                    camera_count += 1

    # 保存JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"模态分配文件创建成功！")
    print(f"{'='*60}")
    print(f"输出文件: {output_path}")
    print(f"总场景数: {total_scenarios}")
    print(f"总智能体数: {total_agents}")
    print(f"  - Ego车辆 (m1/lidar): {ego_count}")
    print(f"  - 协同车辆 (m2/camera): {camera_count}")
    print(f"{'='*60}\n")
    
    # 显示示例
    print("示例场景分配:")
    for i, (scenario_name, cav_dict) in enumerate(scenario_cav_modality_dict.items()):
        if i >= 3:  # 只显示前3个
            break
        print(f"\n场景: {scenario_name}")
        for cav_id, modality in cav_dict.items():
            role = "Ego (Lidar)" if modality == 'm1' else "Collab (Camera)"
            print(f"  - CAV {cav_id}: {modality} ({role})")
    
    print(f"\n... 共 {total_scenarios} 个场景")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "/home/zzh/projects/HEAL/dataset/OPV2V"
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "opencood/logs/heter_modality_assign/opv2v_ego_lidar_others_camera.json"
    
    print(f"数据集路径: {root_dir}")
    print(f"输出路径: {output_path}")
    
    create_ego_lidar_others_camera_assignment(root_dir, output_path)
