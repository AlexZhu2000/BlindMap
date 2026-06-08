#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV特征分布分析工具

purpose: 验证不同大小物体的融合BEV特征是否具有区分性
         支持或反驳"按物体大小分类Proto"的假设

使用方法：
    python analyze_bev_feature_distribution.py \
        --model_dir /path/to/blindmap/model \
        --config_dir /path/to/config \
        --output_dir ./feature_analysis_results \
        --sample_size 200 (optional)
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# BlindMap imports
sys.path.insert(0, '/home/zzh/projects/BlindMap')
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils

# 降维可视化库
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans


class BEVFeatureAnalyzer:
    """分析BEV融合特征的分布与大小相关性"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda', output_dir: str = './results'):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fused_feature_cache = {}  # 用于Hook缓存中间特征
        
        print(f"\n{'='*80}")
        print("BEV Feature Distribution Analyzer 初始化")
        print(f"{'='*80}\n")
        
        # 加载配置
        print("[1/4] 加载配置...")
        self.config = yaml_utils.load_yaml(config_path)
        
        # 提取BEV参数
        self.cav_range = self.config.get('cav_lidar_range', [-102.4, -102.4, -3, 102.4, 102.4, 1])
        self.H = int(self.cav_range[4] - self.cav_range[1])
        self.W = int(self.cav_range[3] - self.cav_range[0])
        print(f"✓ BEV网格: {self.H} × {self.W}")
        
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
        
        # 注册Hook来捕获融合特征
        self._register_hooks()
        
        # 加载数据集
        print("[4/4] 加载数据集...")
        self.dataset = build_dataset(self.config, visualize=False, train=False)
        print(f"✓ 数据集加载: {len(self.dataset)} 样本\n")
        
        # 大小分类阈值
        self.size_thresholds = [0.01, 0.04, 0.12, 0.30]  # 5个类别的边界
        self.size_class_names = ['极小', '小', '中', '大', '超大']
    
    def _register_hooks(self):
        """注册Hook来捕获融合特征"""
        def hook_cls_head(module, input, output):
            # cls_head的输入就是融合特征
            if isinstance(input, tuple) and len(input) > 0:
                feat = input[0]
                if isinstance(feat, torch.Tensor):
                    self.fused_feature_cache['fused_feature'] = feat.detach()
        
        def hook_shrink_conv(module, input, output):
            # 如果有shrink_conv，特征是输出
            if isinstance(output, torch.Tensor):
                self.fused_feature_cache['fused_feature'] = output.detach()
        
        # 优先选择cls_head（在shrink_conv后面）
        if hasattr(self.model, 'cls_head'):
            self.model.cls_head.register_forward_hook(hook_cls_head)
            print("✓ 已注册Hook到cls_head")
        # 备用选项：shrink_conv
        elif hasattr(self.model, 'shrink_conv'):
            self.model.shrink_conv.register_forward_hook(hook_shrink_conv)
            print("✓ 已注册Hook到shrink_conv")
        else:
            print("⚠ 无法找到合适的Hook点！")
        
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
    
    def _to_device(self, data):
        """递归转移到设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            result = [self._to_device(item) for item in data]
            return type(data)(result)
        return data
    
    def compute_size_class(self, bbox_size_x: float, bbox_size_y: float) -> int:
        """计算物体的大小类别"""
        size_ratio = (bbox_size_x / self.W) * (bbox_size_y / self.H)
        
        if size_ratio < self.size_thresholds[0]:
            return 0  # 极小
        elif size_ratio < self.size_thresholds[1]:
            return 1  # 小
        elif size_ratio < self.size_thresholds[2]:
            return 2  # 中
        elif size_ratio < self.size_thresholds[3]:
            return 3  # 大
        else:
            return 4  # 超大
    
    def extract_gt_bbox_features(self, batch_data: Dict, modality_features: Dict) -> List[Dict]:
        """
        基于GT框提取融合特征
        
        Args:
            batch_data: 批数据 (已collate)
            modality_features: 融合特征 shape (1, C, H_fused, W_fused)
        
        Returns:
            List of feature dicts with metadata
        """
        ego_data = batch_data['ego']
        fused_feat = modality_features.get('fused_feature', None)
        
        if fused_feat is None:
            return []
        
        # 获取GT框和mask
        gt_boxes = ego_data.get('object_bbx_center')  # (B, max_objs, 3)
        gt_mask = ego_data.get('object_bbx_mask')      # (B, max_objs)
        
        if gt_boxes is None:
            return []
        
        # 转为numpy
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # 提取第一个样本 (batch_size=1)
        if gt_boxes.ndim == 3:
            gt_boxes = gt_boxes[0]
        if gt_mask.ndim == 2:
            gt_mask = gt_mask[0]
        
        results = []
        
        # 对每个有效GT框处理
        for obj_idx, box in enumerate(gt_boxes):
            if gt_mask[obj_idx] < 0.5:
                continue
            
            # 框坐标
            cx, cy, cz = box[0], box[1], box[2]
            
            # 获取框大小 (假设框有额外的尺寸信息)
            # object_bbx_center通常是 (x, y, z) + 其他信息
            # 需要获得实际的长宽高
            if len(box) >= 6:
                # 假设格式: [x, y, z, length, width, height, ...]
                length, width = box[3], box[4]
            else:
                # 如果没有尺寸信息，跳过
                continue
            
            # 转换到BEV像素坐标
            px_center = (cx - self.cav_range[0]) / (self.cav_range[3] - self.cav_range[0]) * self.W
            py_center = (cy - self.cav_range[1]) / (self.cav_range[4] - self.cav_range[1]) * self.H
            
            px_size = (length / (self.cav_range[3] - self.cav_range[0])) * self.W
            py_size = (width / (self.cav_range[4] - self.cav_range[1])) * self.H
            
            px_size = max(1, int(px_size))  # 确保至少1像素
            py_size = max(1, int(py_size))
            
            # 计算大小类别
            size_ratio = (length / (self.cav_range[3] - self.cav_range[0])) * \
                        (width / (self.cav_range[4] - self.cav_range[1]))
            size_class = self.compute_size_class(px_size, py_size)
            
            # 创建物体的mask
            x_start = max(0, int(px_center - px_size / 2))
            x_end = min(self.W, int(px_center + px_size / 2) + 1)
            y_start = max(0, int(py_center - py_size / 2))
            y_end = min(self.H, int(py_center + py_size / 2) + 1)
            
            # 提取特征
            if x_end > x_start and y_end > y_start:
                # fused_feat shape: (1, C, H, W)
                roi_feat = fused_feat[0, :, y_start:y_end, x_start:x_end]  # (C, h, w)
                
                # 聚合特征: 使用mean pooling
                feat_aggregated = roi_feat.mean(dim=[1, 2])  # (C,)
                
                results.append({
                    'feature': feat_aggregated.cpu().detach().numpy(),
                    'size_class': size_class,
                    'size_ratio': size_ratio,
                    'bbox_size': (px_size, py_size),
                    'num_pixels': (x_end - x_start) * (y_end - y_start),
                    'center': (px_center, py_center),
                    'gt_box_id': obj_idx
                })
        
        return results
    
    def process_batch(self, batch_data: Dict) -> Dict:
        """前向推理获取融合特征"""
        ego_data = self._to_device(batch_data['ego'])
        
        # 清空缓存
        self.fused_feature_cache.clear()
        
        try:
            with torch.no_grad():
                output_dict = self.model(ego_data)
            
            # Hook已经捕获了融合特征
            if 'fused_feature' in self.fused_feature_cache:
                output_dict['fused_feature'] = self.fused_feature_cache['fused_feature']
            elif 'spatial_features_2d' in output_dict:
                # 备用方案：使用spatial_features_2d
                output_dict['fused_feature'] = output_dict['spatial_features_2d']
            else:
                # 最后尝试从任何包含"feature"的键中获取
                for key in output_dict.keys():
                    if isinstance(output_dict[key], torch.Tensor) and output_dict[key].ndim == 4:
                        output_dict['fused_feature'] = output_dict[key]
                        break
            
            return output_dict
        except RuntimeError as e:
            # 处理CUDA相关错误
            import traceback
            print(f"⚠ 前向推理出错: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            raise
    
    def analyze_dataset(self, num_samples: int = None, sample_indices: List[int] = None):
        """
        分析整个数据集的特征分布
        
        Args:
            num_samples: 处理的样本数量
            sample_indices: 特定的样本索引列表
        """
        print(f"\n{'='*80}")
        print("[分析] 遍历数据集提取特征")
        print(f"{'='*80}\n")
        
        # 准备数据加载器
        if sample_indices is not None:
            dataset_subset = Subset(self.dataset, sample_indices)
            num_samples = len(sample_indices)
        elif num_samples is not None:
            dataset_subset = Subset(self.dataset, range(min(num_samples, len(self.dataset))))
        else:
            dataset_subset = self.dataset
            num_samples = len(self.dataset)
        
        dataloader = DataLoader(
            dataset_subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_batch_train
        )
        
        # 收集特征
        all_features = []
        grouped_features = defaultdict(list)
        metadata = []
        
        for i, batch_data in enumerate(tqdm(dataloader, total=len(dataloader))):
            try:
                # 前向推理
                output_dict = self.process_batch(batch_data)
                
                # 获取融合特征
                fused_feature = output_dict.get('fused_feature', None)
                
                if fused_feature is None:
                    print(f"  ⚠ 样本{i}: 未找到融合特征")
                    continue
                
                # 准备modality_features dict
                modality_features = {'fused_feature': fused_feature}
                
                # 提取GT框对应的特征
                sample_features = self.extract_gt_bbox_features(batch_data, modality_features)
                
                if not sample_features:
                    continue
                
                # 收集结果
                for feat_info in sample_features:
                    all_features.append(feat_info)
                    size_class = feat_info['size_class']
                    grouped_features[size_class].append(feat_info)
            
            except Exception as e:
                print(f"  ❌ 样本{i} 处理失败: {str(e)[:100]}")
                if i < 3:  # 只打印前3个错误的详细信息
                    import traceback
                    print(f"     {traceback.format_exc()[:500]}")
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
    
    def compute_clustering_metrics(self, all_features: List[Dict]) -> Dict:
        """计算聚类指标"""
        print(f"\n{'='*80}")
        print("[指标] 计算聚类质量指标")
        print(f"{'='*80}\n")
        
        # 准备特征矩阵和标签
        X = np.array([f['feature'] for f in all_features])  # (N, C)
        y = np.array([f['size_class'] for f in all_features])  # (N,)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 计算指标
        metrics = {}
        
        # Silhouette Score
        sil_score = silhouette_score(X_scaled, y)
        metrics['silhouette_score_global'] = float(sil_score)
        print(f"✓ Silhouette Score (全局): {sil_score:.4f}")
        
        # Davies-Bouldin Index
        db_score = davies_bouldin_score(X_scaled, y)
        metrics['davies_bouldin_index'] = float(db_score)
        print(f"✓ Davies-Bouldin Index: {db_score:.4f}")
        
        # Calinski-Harabasz Index
        ch_score = calinski_harabasz_score(X_scaled, y)
        metrics['calinski_harabasz_index'] = float(ch_score)
        print(f"✓ Calinski-Harabasz Index: {ch_score:.4f}")
        
        # 按类别计算Silhouette
        silhouette_per_class = {}
        for size_class in range(5):
            mask = y == size_class
            if mask.sum() > 1:
                sil = silhouette_score(X_scaled[mask], y[mask])
                silhouette_per_class[size_class] = float(sil)
                name = self.size_class_names[size_class]
                print(f"  - Class {size_class} ({name}): {sil:.4f}")
            else:
                silhouette_per_class[size_class] = None
        
        metrics['silhouette_score_per_class'] = silhouette_per_class
        
        # 类内方差和类间距离
        class_stats = {}
        for size_class in range(5):
            mask = y == size_class
            if mask.sum() > 0:
                class_feat = X_scaled[mask]
                center = class_feat.mean(axis=0)
                intra_dist = np.mean(np.linalg.norm(class_feat - center, axis=1))
                class_stats[size_class] = {
                    'center': center.tolist(),
                    'intra_distance': float(intra_dist),
                    'num_samples': int(mask.sum())
                }
        
        metrics['class_statistics'] = class_stats
        
        # 类间距离
        inter_distances = {}
        for i in range(5):
            for j in range(i+1, 5):
                if i in class_stats and j in class_stats:
                    dist = np.linalg.norm(
                        np.array(class_stats[i]['center']) - 
                        np.array(class_stats[j]['center'])
                    )
                    inter_distances[f"{i}-{j}"] = float(dist)
        
        metrics['inter_class_distances'] = inter_distances
        
        return metrics, X_scaled, y
    
    def visualize_with_umap(self, X_scaled: np.ndarray, y: np.ndarray, output_file: str):
        """使用UMAP降维可视化"""
        if not UMAP_AVAILABLE:
            print("⚠ UMAP不可用，跳过可视化")
            return
        
        print(f"\n[可视化] UMAP降维...")
        
        # UMAP降维
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        # 绘制
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # 全局散点图
        for size_class in range(5):
            mask = y == size_class
            name = self.size_class_names[size_class]
            ax1.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=colors[size_class], label=name, s=30, alpha=0.7)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('BEV Feature Distribution (All Classes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 按类别单独显示
        for size_class in range(5):
            mask = y == size_class
            if mask.sum() > 0:
                ax2.scatter(X_umap[mask, 0], X_umap[mask, 1],
                           c=colors[size_class], label=self.size_class_names[size_class],
                           s=30, alpha=0.7)
        
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('BEV Feature Distribution (Color-coded by Size Class)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {output_file}")
        plt.close()
    
    def visualize_metrics(self, metrics: Dict, output_file: str):
        """可视化聚类指标"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Silhouette scores per class
        silhouette_per_class = metrics['silhouette_score_per_class']
        classes = [self.size_class_names[i] for i in range(5)]
        scores = [silhouette_per_class.get(i, 0) for i in range(5)]
        
        axes[0].bar(classes, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score per Size Class')
        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Class statistics
        class_stats = metrics['class_statistics']
        intra_dists = [class_stats[i]['intra_distance'] if i in class_stats else 0 for i in range(5)]
        sample_counts = [class_stats[i]['num_samples'] if i in class_stats else 0 for i in range(5)]
        
        ax2_twin = axes[1].twinx()
        
        bars = axes[1].bar(classes, intra_dists, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
                          alpha=0.6, label='Intra-class Distance')
        ax2_twin.plot(classes, sample_counts, 'ko-', linewidth=2, markersize=8, label='Sample Count')
        
        axes[1].set_ylabel('Intra-class Distance')
        ax2_twin.set_ylabel('Number of Samples')
        axes[1].set_title('Class Homogeneity and Sample Counts')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {output_file}")
        plt.close()
    
    def generate_report(self, all_features: List[Dict], grouped_features: Dict, metrics: Dict):
        """生成分析报告"""
        print(f"\n{'='*80}")
        print("[报告] 生成分析总结")
        print(f"{'='*80}\n")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'num_total_features': len(all_features),
                'num_size_classes': 5,
                'size_thresholds': self.size_thresholds,
                'bev_size': (self.W, self.H)
            },
            'feature_distribution': {
                self.size_class_names[i]: len(grouped_features[i])
                for i in range(5)
            },
            'metrics': metrics,
            'conclusions': self._draw_conclusions(metrics)
        }
        
        # 保存为JSON
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ 报告已保存: {report_path}")
        
        # 打印总结
        print("\n" + "="*80)
        print("分析总结")
        print("="*80)
        for conclusion in report['conclusions']:
            print(f"  {conclusion}")
        
        return report
    
    def _draw_conclusions(self, metrics: Dict) -> List[str]:
        """根据指标绘制结论"""
        conclusions = []
        
        sil_global = metrics['silhouette_score_global']
        db_global = metrics['davies_bouldin_index']
        
        conclusions.append(f"全局Silhouette Score: {sil_global:.4f}")
        conclusions.append(f"Davies-Bouldin Index: {db_global:.4f}")
        
        if sil_global > 0.4:
            conclusions.append("✅ 聚类效果优秀: 不同大小类别特征分离明显")
            conclusions.append("✅ Proto大小分类方案可行")
        elif sil_global > 0.2:
            conclusions.append("⚠️ 聚类效果一般: 部分类别有重叠")
            conclusions.append("⚠️ 可能需要调整大小阈值或异常处理")
        else:
            conclusions.append("❌ 聚类效果差: 类别间无明显分离")
            conclusions.append("❌ 需要重新考虑分类维度或整体方案")
        
        return conclusions
    
    def run(self, num_samples: int = None):
        """运行完整分析"""
        # 提取特征
        all_features, grouped_features = self.analyze_dataset(num_samples=num_samples)
        
        if len(all_features) == 0:
            print("❌ 未提取到特征，分析失败")
            return
        
        # 计算指标
        metrics, X_scaled, y = self.compute_clustering_metrics(all_features)
        
        # 可视化
        print(f"\n{'='*80}")
        print("[可视化] 生成图表")
        print(f"{'='*80}\n")
        
        self.visualize_with_umap(X_scaled, y, str(self.output_dir / 'umap_visualization.png'))
        self.visualize_metrics(metrics, str(self.output_dir / 'metrics_visualization.png'))
        
        # 生成报告
        report = self.generate_report(all_features, grouped_features, metrics)
        
        # 保存特征
        np.save(self.output_dir / 'features_raw.npy', X_scaled)
        np.save(self.output_dir / 'labels.npy', y)
        
        print(f"\n{'='*80}")
        print(f"✓ 分析完成，结果保存到: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='BEV Feature Distribution Analysis')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to BlindMap model directory')
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./bev_feature_analysis',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = BEVFeatureAnalyzer(
        model_path=args.model_dir,
        config_path=args.config_file,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 运行分析
    analyzer.run(num_samples=args.num_samples)


if __name__ == '__main__':
    main()
