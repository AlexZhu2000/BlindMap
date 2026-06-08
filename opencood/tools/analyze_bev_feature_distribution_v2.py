#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV特征分布分析工具 - 改进版本

基于已验证的SelfDistill proto_comparison_simple.py
使用标准数据加载方式确保兼容性

使用示例：
    python analyze_bev_feature_distribution_v2.py \
        --model /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/net_epoch37.pth \
        --config /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/config.yaml \
        --output /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/proto_results_v2 \
        --max_samples 100
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/home/zzh/projects/BlindMap')

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
except ImportError as e:
    print(f"Warning: Missing dependency {e}")


class BEVFeatureAnalyzerV2:
    """改进的BEV特征分布分析器"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda', output_dir: str = './results'):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("[初始化] BEV特征分析器")
        print(f"{'='*80}\n")
        
        # 加载配置
        print("[1/4] 加载配置...")
        self.config = yaml_utils.load_yaml(config_path)
        
        # 提取BEV参数
        self.cav_range = self.config.get('cav_lidar_range', [-102.4, -102.4, -3, 102.4, 102.4, 1])
        self.H = int(self.cav_range[4] - self.cav_range[1])
        self.W = int(self.cav_range[3] - self.cav_range[0])
        print(f"✓ BEV网格: {self.W} × {self.H}")
        
        # 创建模型
        print("[2/4] 创建模型...")
        self.model = train_utils.create_model(self.config)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ 模型: {self.config['model']['core_method']}")
        
        # 加载权重
        print("[3/4] 加载预训练权重...")
        self._load_checkpoint(model_path)
        print(f"✓ 权重加载成功")
        
        # 加载数据集
        print("[4/4] 加载数据集...")
        self.dataset = build_dataset(self.config, visualize=False, train=False)
        print(f"✓ 数据集加载: {len(self.dataset)} 样本\n")
        
        # 大小分类阈值
        self.size_thresholds = [0.01, 0.04, 0.12, 0.30]
        self.size_class_names = ['极小', '小', '中', '大', '超大']
        
        # Hook缓存
        self.fused_feature_cache = None
        self._register_hooks()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 处理DDP前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
    
    def _register_hooks(self):
        """注册Hook捕获融合特征"""
        def capture_fused_feature(module, input, output):
            # 捕获cls_head的输入（融合特征）
            if isinstance(input, tuple) and len(input) > 0:
                feat = input[0]
                if isinstance(feat, torch.Tensor):
                    self.fused_feature_cache = feat.detach()
        
        if hasattr(self.model, 'cls_head'):
            self.model.cls_head.register_forward_hook(capture_fused_feature)
            print("✓ Hook已注册到cls_head")
    
    def compute_size_class(self, bbox_size_x: float, bbox_size_y: float) -> int:
        """计算物体大小类别"""
        size_ratio = (bbox_size_x / self.W) * (bbox_size_y / self.H)
        
        if size_ratio < self.size_thresholds[0]:
            return 0
        elif size_ratio < self.size_thresholds[1]:
            return 1
        elif size_ratio < self.size_thresholds[2]:
            return 2
        elif size_ratio < self.size_thresholds[3]:
            return 3
        else:
            return 4
    
    def extract_gt_bbox_features(self, batch_data: dict, fused_feature: torch.Tensor) -> list:
        """基于GT框提取BEV特征"""
        ego_data = batch_data['ego']
        
        # 获取GT信息
        gt_boxes = ego_data.get('object_bbx_center')
        gt_mask = ego_data.get('object_bbx_mask')
        
        if gt_boxes is None or fused_feature is None:
            return []
        
        # 转numpy
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # 处理batch维度
        if gt_boxes.ndim == 3:
            gt_boxes = gt_boxes[0]
        if gt_mask.ndim == 2:
            gt_mask = gt_mask[0]
        
        results = []
        
        # 遍历所有GT框
        for obj_idx, box in enumerate(gt_boxes):
            if gt_mask[obj_idx] < 0.5:
                continue
            
            # 解析框信息
            cx, cy, cz = box[0], box[1], box[2]
            
            # 获取框大小
            if len(box) < 6:
                continue
            
            length, width = box[3], box[4]
            
            # 转到像素坐标
            px_center = (cx - self.cav_range[0]) / (self.cav_range[3] - self.cav_range[0]) * self.W
            py_center = (cy - self.cav_range[1]) / (self.cav_range[4] - self.cav_range[1]) * self.H
            
            px_size = max(1, int((length / (self.cav_range[3] - self.cav_range[0])) * self.W))
            py_size = max(1, int((width / (self.cav_range[4] - self.cav_range[1])) * self.H))
            
            # 计算大小类别
            size_ratio = ((length / (self.cav_range[3] - self.cav_range[0])) * 
                         (width / (self.cav_range[4] - self.cav_range[1])))
            size_class = self.compute_size_class(px_size, py_size)
            
            # 提取ROI
            x_start = max(0, int(px_center - px_size / 2))
            x_end = min(self.W, int(px_center + px_size / 2) + 1)
            y_start = max(0, int(py_center - py_size / 2))
            y_end = min(self.H, int(py_center + py_size / 2) + 1)
            
            if x_end > x_start and y_end > y_start:
                # 提取特征
                roi_feat = fused_feature[0, :, y_start:y_end, x_start:x_end]
                feat_vec = roi_feat.mean(dim=[1, 2]).cpu().detach().numpy()
                
                results.append({
                    'feature': feat_vec,
                    'size_class': size_class,
                    'size_ratio': float(size_ratio),
                    'bbox_size': (px_size, py_size),
                    'num_pixels': (x_end - x_start) * (y_end - y_start)
                })
        
        return results
    
    def analyze_dataset(self, num_samples: int = None):
        """分析整个数据集"""
        print(f"\n{'='*80}")
        print("[分析] 遍历数据集...")
        print(f"{'='*80}\n")
        
        # 准备数据加载器
        if num_samples is not None:
            subset = Subset(self.dataset, range(min(num_samples, len(self.dataset))))
        else:
            subset = self.dataset
        
        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_batch_train
        )
        
        all_features = []
        grouped_features = defaultdict(list)
        
        for i, batch_data in enumerate(tqdm(dataloader)):
            try:
                # 前向推理
                with torch.no_grad():
                    # 递归转移所有数据到设备
                    def to_device(data, device):
                        if isinstance(data, torch.Tensor):
                            return data.to(device)
                        elif isinstance(data, dict):
                            return {k: to_device(v, device) for k, v in data.items()}
                        elif isinstance(data, (list, tuple)):
                            result = [to_device(item, device) for item in data]
                            return type(data)(result)
                        return data
                    
                    ego_data = to_device(batch_data['ego'], self.device)
                    self.fused_feature_cache = None
                    _ = self.model(ego_data)
                
                # 获取融合特征
                if self.fused_feature_cache is None:
                    continue
                
                # 提取GT箱体特征
                features = self.extract_gt_bbox_features(batch_data, self.fused_feature_cache)
                
                for feat_info in features:
                    all_features.append(feat_info)
                    grouped_features[feat_info['size_class']].append(feat_info)
            
            except Exception as e:
                print(f"  ⚠ 样本{i}: {str(e)[:80]}")
                continue
        
        print(f"\n{'='*80}")
        print("✓ 特征提取完成")
        print(f"{'='*80}")
        print(f"总特征数: {len(all_features)}")
        for size_class in range(5):
            count = len(grouped_features[size_class])
            name = self.size_class_names[size_class]
            print(f"  Class {size_class} ({name}): {count} 特征")
        
        return all_features, grouped_features
    
    def compute_metrics(self, all_features: list) -> dict:
        """计算聚类指标"""
        print(f"\n{'='*80}")
        print("[评估] 计算指标...")
        print(f"{'='*80}\n")
        
        X = np.array([f['feature'] for f in all_features])
        y = np.array([f['size_class'] for f in all_features])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 计算指标
        metrics = {}
        
        sil = silhouette_score(X_scaled, y)
        db = davies_bouldin_score(X_scaled, y)
        ch = calinski_harabasz_score(X_scaled, y)
        
        print(f"✓ Silhouette Score: {sil:.4f}")
        print(f"✓ Davies-Bouldin Index: {db:.4f}")
        print(f"✓ Calinski-Harabasz Index: {ch:.4f}")
        
        metrics['silhouette'] = float(sil)
        metrics['davies_bouldin'] = float(db)
        metrics['calinski_harabasz'] = float(ch)
        
        return metrics, X_scaled, y
    
    def visualize(self, X_scaled: np.ndarray, y: np.ndarray):
        """可视化特征分布"""
        if not UMAP_AVAILABLE:
            print("⚠ UMAP不可用，跳过可视化")
            return
        
        print(f"\n{'='*80}")
        print("[可视化] UMAP降维...")
        print(f"{'='*80}\n")
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for size_class in range(5):
            mask = y == size_class
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=colors[size_class], label=self.size_class_names[size_class],
                      s=30, alpha=0.7)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('BEV Feature Distribution by Size Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'umap_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {output_path}")
        plt.close()
    
    def run(self, num_samples: int = None):
        """运行完整分析"""
        all_features, grouped = self.analyze_dataset(num_samples=num_samples)
        
        if len(all_features) == 0:
            print("❌ 未提取到特征")
            return
        
        metrics, X_scaled, y = self.compute_metrics(all_features)
        self.visualize(X_scaled, y)
        
        # 保存结果
        report = {
            'total_features': len(all_features),
            'class_distribution': {str(i): len(grouped[i]) for i in range(5)},
            'metrics': metrics
        }
        
        report_path = self.output_dir / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ 报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='./proto_analysis_results')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    analyzer = BEVFeatureAnalyzerV2(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        output_dir=args.output
    )
    
    analyzer.run(num_samples=args.max_samples)


if __name__ == '__main__':
    main()
