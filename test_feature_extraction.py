#!/usr/bin/env python3
"""快速测试: 修改后的特征提取是否工作"""

import sys
import os
sys.path.insert(0, '/home/zzh/projects/BlindMap')

# 修复cuDNN问题
import torch
torch.backends.cudnn.enabled = False

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

# 找最新的模型
log_dir = '/home/zzh/projects/BlindMap/opencood/logs'
subdirs = [d for d in os.listdir(log_dir) if d.startswith('BlindMap_opv2v')]
subdirs.sort()
latest_dir = os.path.join(log_dir, subdirs[-1])

config_path = os.path.join(latest_dir, 'config.yaml')
model_path = os.path.join(latest_dir, 'net_epoch37.pth')

print(f"最新模型目录: {latest_dir}")

# 手动初始化模型（不使用完整的BEVFeatureAnalyzer）
print("\n[初始化] 加载配置和模型...")
try:
    config = yaml_utils.load_yaml(config_path)
    model = train_utils.create_model(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.cuda().eval()
    print("✓ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 初始化数据集
print("\n[初始化] 加载数据集...")
try:
    dataset = build_dataset(config, visualize=False, train=False)
    print(f"✓ 数据集加载成功: {len(dataset)} 样本")
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    sys.exit(1)

# 测试前向推理
print("\n[测试] 前向推理...")
try:
    batch = dataset.collate_batch_train([dataset[0]])
    
    # 递归转移所有张量到GPU
    def to_device(data, device='cuda'):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            result = [to_device(item, device) for item in data]
            return type(data)(result)
        return data
    
    ego_data = to_device(batch['ego'], 'cuda')
    
    # 捕获融合特征
    fused_features = {}
    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            feat = input[0]
            if isinstance(feat, torch.Tensor):
                fused_features['fused_feature'] = feat
    
    # 为cls_head注册hook
    if hasattr(model, 'cls_head'):
        model.cls_head.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output_dict = model(ego_data)
    
    if 'fused_feature' in fused_features:
        fused_feat = fused_features['fused_feature']
        print(f"✓ 融合特征获取成功")
        print(f"  Shape: {fused_feat.shape}")
        print(f"  Dtype: {fused_feat.dtype}")
    else:
        print(f"❌ 融合特征未捕获")
except Exception as e:
    print(f"❌ 前向推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 测试完成")
