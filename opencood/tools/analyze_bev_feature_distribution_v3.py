#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV特征分布分析工具 v3 - 按object组织，支持yaw角旋转处理

新思路：对每个检测目标提取其覆盖区域的融合特征，然后用2D可视化展示
- 一个样本：m个检测目标 -> m个特征向量 (不同颜色)
- 使用UMAP/PCA降维到2D
- 同一object的特征用同一颜色表示，观察特征聚集程度

改进点：
- 考虑7维bbox中的yaw角，精确计算旋转后的ROI区域
- 支持两种方法：gt_bbox (基于GT框) 和 pos_mask (基于模型输出的pos_equal_one)
- 记录和统计yaw角度信息

使用示例（方法1：GT bbox with yaw）:
    python analyze_bev_feature_distribution_v3.py \
        --model /path/to/model.pth \
        --config /path/to/config.yaml \
        --output ./results \
        --max_samples 50 \
        --extract_method gt_bbox

使用示例（方法2：pos_mask，基于模型的OCC输出 - 不推荐）:
    python analyze_bev_feature_distribution_v3.py \
        --model /path/to/model.pth \
        --config /path/to/config.yaml \
        --output ./results \
        --max_samples 50 \
        --extract_method pos_mask

使用示例（方法3：pred_bbox，基于模型的BBox预测 - 推荐！）:
    python analyze_bev_feature_distribution_v3.py \
        --model /path/to/model.pth \
        --config /path/to/config.yaml \
        --output ./results \
        --max_samples 50 \
        --extract_method pred_bbox
        
三种方法的对比：
- gt_bbox: 使用GT标注框，包含yaw角旋转，精确但依赖GT标注（离线分析用）
- pos_mask: 使用模型输出的OCC热力图（已弃用，OCC只是辅助信息）
- pred_bbox: 使用模型最终预测的BBox（推荐！直接体现模型的特征理解）
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

sys.path.insert(0, '/home/zzh/projects/BlindMap')

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.patches as mpatches
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
except ImportError as e:
    print(f"Warning: Missing dependency {e}")


class BEVFeatureAnalyzerV3:
    """按object组织的BEV特征分析器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda',
        output_dir: str = './results',
        enable_background: bool = True,
        bg_points_per_sample: int = 1000,
        bg_seed: int = 42,
        interactive_3d: bool = True,
        object_feature_mode: str = 'all_pixels',
        vis_color_mode: str = 'multi_object',
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_background = enable_background
        self.bg_points_per_sample = max(0, int(bg_points_per_sample))
        self.bg_rng = np.random.default_rng(bg_seed)
        self.interactive_3d = interactive_3d
        self.object_feature_mode = object_feature_mode
        self.vis_color_mode = vis_color_mode
        
        print(f"\n{'='*80}")
        print("[初始化] BEV特征分析器 v3 (按Object组织)")
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
        
        # Hook缓存
        self.fused_feature_cache = None
        self.occ_map_list_cache = None
        self.model_output_cache = None
        self._register_hooks()

    def _extract_random_background_features(self, fused_feature: torch.Tensor, objects: list) -> dict:
        """随机采样背景像素特征（避开object包围框）"""
        if fused_feature is None or self.bg_points_per_sample <= 0:
            return None

        valid_objects = [obj for obj in objects if obj.get('object_id', -1) >= 0]

        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        for obj in valid_objects:
            cx, cy = obj['center']
            sx, sy = obj['size']
            x_min = max(0, int(np.floor(cx - sx / 2.0)))
            x_max = min(self.W, int(np.ceil(cx + sx / 2.0)) + 1)
            y_min = max(0, int(np.floor(cy - sy / 2.0)))
            y_max = min(self.H, int(np.ceil(cy + sy / 2.0)) + 1)
            if x_max > x_min and y_max > y_min:
                occ_mask[y_min:y_max, x_min:x_max] = True

        bg_candidates = np.argwhere(~occ_mask)
        if len(bg_candidates) == 0:
            return None

        sample_num = min(self.bg_points_per_sample, len(bg_candidates))
        select_idx = self.bg_rng.choice(len(bg_candidates), size=sample_num, replace=False)
        selected = bg_candidates[select_idx]  # (K, 2) -> [y, x]

        ys = selected[:, 0]
        xs = selected[:, 1]
        bg_feat = fused_feature[0, :, ys, xs].permute(1, 0).cpu().detach().numpy()  # (K, C)

        return {
            'features': bg_feat,
            'object_id': -1,
            'center': (-1.0, -1.0),
            'size': (0.0, 0.0),
            'yaw': 0.0,
            'size_ratio': 0.0,
            'num_pixels': int(sample_num),
            'method': 'background',
            'is_background': True,
        }
    
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
            if isinstance(input, tuple) and len(input) > 0:
                feat = input[0]
                if isinstance(feat, torch.Tensor):
                    self.fused_feature_cache = feat.detach()
        
        if hasattr(self.model, 'cls_head'):
            self.model.cls_head.register_forward_hook(capture_fused_feature)
            print("✓ Hook已注册到cls_head")
    
    def extract_object_features(self, batch_data: dict, fused_feature: torch.Tensor, method: str = 'gt_bbox') -> list:
        """
        对每个object提取特征
        
        Args:
            batch_data: 批数据
            fused_feature: 融合特征 (1, C, H, W)
            method: 'gt_bbox' - 使用GT bbox; 
                   'pos_mask' - 使用occ掩码（已弃用）;
                   'pred_bbox' - 使用模型预测的BBox（推荐！）
        
        返回: List of {
            'feature': 特征向量,
            'object_id': 物体索引,
            ...
        }
        """
        if method == 'gt_bbox':
            ego_data = batch_data['ego']
            return self._extract_by_gt_bbox(ego_data, fused_feature)
        elif method == 'pos_mask':
            return self._extract_by_pos_mask(fused_feature)
        elif method == 'pred_bbox':
            return self._extract_by_pred_bbox(fused_feature)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_by_gt_bbox(self, ego_data: dict, fused_feature: torch.Tensor) -> list:
        """
        基于GT bbox获取ROI，考虑yaw角旋转
        
        bbox格式: (x, y, z, l, w, h, yaw) 其中l=length, w=width
        """
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
        object_id = 0
        
        # 遍历所有GT框
        for obj_idx, box in enumerate(gt_boxes):
            if gt_mask[obj_idx] < 0.5:
                continue
            
            # 解析7维框信息: (x, y, z, l, w, h, yaw)
            cx, cy, cz = box[0], box[1], box[2]
            length, width, height = box[3], box[4], box[5]
            yaw = box[6] if len(box) > 6 else 0.0
            
            # 转到像素坐标
            px_center = (cx - self.cav_range[0]) / (self.cav_range[3] - self.cav_range[0]) * self.W
            py_center = (cy - self.cav_range[1]) / (self.cav_range[4] - self.cav_range[1]) * self.H
            
            # 计算旋转后的bbox像素大小
            px_length = (length / (self.cav_range[3] - self.cav_range[0])) * self.W
            px_width = (width / (self.cav_range[4] - self.cav_range[1])) * self.H
            
            # 获取旋转框的四个顶点（考虑yaw角）
            vertices = self._get_rotated_bbox_vertices(px_center, py_center, px_length, px_width, yaw)
            
            # 获取bbox包围框
            x_min = max(0, int(np.floor(np.min(vertices[:, 0]))))
            x_max = min(self.W, int(np.ceil(np.max(vertices[:, 0]))) + 1)
            y_min = max(0, int(np.floor(np.min(vertices[:, 1]))))
            y_max = min(self.H, int(np.ceil(np.max(vertices[:, 1]))) + 1)
            
            if x_max > x_min and y_max > y_min:
                # 提取ROI特征（保存所有像素，不取均值）
                roi_feat = fused_feature[0, :, y_min:y_max, x_min:x_max]  # (C, h, w)
                
                # 将特征转为 (num_pixels, C) 形状
                C = roi_feat.shape[0]
                feat_spatial = roi_feat.cpu().detach().numpy()  # (C, h, w)
                feat_spatial = np.transpose(feat_spatial, (1, 2, 0))  # (h, w, C)
                num_pixels = feat_spatial.shape[0] * feat_spatial.shape[1]
                feat_array = feat_spatial.reshape(-1, C)  # (num_pixels, C)
                
                size_ratio = (length / (self.cav_range[3] - self.cav_range[0])) * \
                           (width / (self.cav_range[4] - self.cav_range[1]))
                
                results.append({
                    'features': feat_array,  # (num_pixels, C) - 所有像素的特征
                    'object_id': object_id,
                    'center': (px_center, py_center),
                    'size': (px_length, px_width),
                    'yaw': float(yaw),
                    'size_ratio': float(size_ratio),
                    'num_pixels': num_pixels,
                    'method': 'gt_bbox'
                })
                
                object_id += 1
        
        return results
    
    def _get_rotated_bbox_vertices(self, cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
        """
        获取旋转后的bbox四个顶点
        
        Args:
            cx, cy: 中心坐标（像素）
            length: 长度（像素）
            width: 宽度（像素）
            yaw: 旋转角（弧度）
        
        Returns:
            vertices: (4, 2) 四个顶点的坐标
        """
        # 计算未旋转的四个顶点（相对于中心）
        half_length = length / 2
        half_width = width / 2
        
        # 本地坐标系中的四个顶点
        local_vertices = np.array([
            [-half_length, -half_width],  # 左后
            [half_length, -half_width],   # 右后
            [half_length, half_width],    # 右前
            [-half_length, half_width]    # 左前
        ])
        
        # 旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # 旋转顶点
        rotated_vertices = local_vertices @ rot_matrix.T
        
        # 平移到全局坐标
        global_vertices = rotated_vertices + np.array([cx, cy])
        
        return global_vertices
    
    def _extract_by_pos_mask(self, fused_feature: torch.Tensor) -> list:
        """
        基于occ_map_list的前景掩码获取ROI
        
        原理：
        - occ_map_list 是模型的多层金字塔输出（不同分辨率）
        - 选择最高分辨率层（Layer 0）的occ预测
        - 使用sigmoid激活后的值作为占用概率
        - 应用阈值获取前景像素，将同一连通部分视为一个object
        
        对应关系：
        - Layer 0 (最高分): 用于obj提取
        - occ_map: (1, 1, H0, W0) sigmoid后∈[0,1]
        """
        if self.occ_map_list_cache is None:
            # print("⚠ occ_map_list_cache is None")
            return []
        
        if len(self.occ_map_list_cache) == 0:
            # print("⚠ occ_map_list_cache is empty")
            return []
        
        # 获取最高分辨率层的occ预测（Layer 0）
        occ_map = self.occ_map_list_cache[0]  # (1, 1, H, W)
        
        # 应用sigmoid激活获取概率
        occ_prob = torch.sigmoid(occ_map)  # (1, 1, H, W)
        
        # 应用阈值获取前景掩码
        threshold = 0.5
        fg_mask = (occ_prob >= threshold).squeeze().cpu().numpy()  # (H, W) 布尔数组
        
        if not fg_mask.any():
            # print(f"⚠ 没有前景像素 (threshold={threshold}, max_prob={occ_prob.max():.4f})")
            return []
        
        # 连通分量标记，识别分属的object
        from scipy.ndimage import label as ndimage_label
        labeled_array, num_features = ndimage_label(fg_mask)
        
        # print(f"ℹ pos_mask: 检测到 {num_features} 个连通部分")
        
        results = []
        H, W = fg_mask.shape
        
        # 遍历每个连通部分（每个是一个detected object）
        for obj_id in range(1, num_features + 1):
            obj_mask = (labeled_array == obj_id)  # (H, W) 布尔数组
            
            if not obj_mask.any():
                continue
            
            # 获取该object的像素坐标
            obj_y, obj_x = np.where(obj_mask)
            
            if len(obj_x) == 0:
                continue
            
            # 从fused_feature中提取该object对应的特征
            y_min, y_max = obj_y.min(), obj_y.max() + 1
            x_min, x_max = obj_x.min(), obj_x.max() + 1
            
            # 提取ROI特征
            roi_feat = fused_feature[0, :, y_min:y_max, x_min:x_max]  # (C, h, w)
            
            # 方式1: 仅对前景像素进行mean pooling
            roi_mask = obj_mask[y_min:y_max, x_min:x_max]  # (h, w)
            roi_mask_tensor = torch.from_numpy(roi_mask).float().to(self.device)
            
            # 使用掩码加权聚合
            roi_feat_masked = roi_feat * roi_mask_tensor
            num_fg_pixels = roi_mask_tensor.sum()
            
            if num_fg_pixels > 0:
                feat_vec = (roi_feat_masked.sum(dim=[1, 2]) / num_fg_pixels).cpu().detach().numpy()
            else:
                feat_vec = roi_feat.mean(dim=[1, 2]).cpu().detach().numpy()
            
            # 计算object的stats
            cx = (obj_x.max() + obj_x.min()) / 2.0
            cy = (obj_y.max() + obj_y.min()) / 2.0
            obj_height = y_max - y_min
            obj_width = x_max - x_min
            
            results.append({
                'feature': feat_vec,
                'object_id': obj_id - 1,  # 从0开始
                'center': (cx, cy),
                'size': (obj_width, obj_height),
                'num_pixels': len(obj_x),  # 前景像素数
                'fg_pixel_ratio': float(len(obj_x)) / ((y_max - y_min) * (x_max - x_min)),
                'occ_prob_mean': float(occ_prob[0, 0, obj_y, obj_x].mean().item()),
                'method': 'pos_mask'
            })
        
        return results
    
    def _extract_by_pred_bbox(self, fused_feature: torch.Tensor) -> list:
        """
        基于模型预测的BBox获取ROI（推荐方法）
        
        原理：
        - 使用模型的cls_preds和reg_preds
        - 应用post_process逻辑生成预测的3D BBox
        - 对每个BBox的ROI区域提取特征
        - 这样可以直接验证"模型认为有目标的区域"对应的特征分布
        
        对应关系：
        - cls_preds: (B, C, H, W) 分类预测
        - reg_preds: (B, 7*anchor_num, H, W) 回归预测
        - 最终输出: 模型预测的3D BBox列表
        """
        if self.model_output_cache is None:
            return []
        
        output_dict = self.model_output_cache
        
        # 获取必要的预测输出
        if 'cls_preds' not in output_dict or 'reg_preds' not in output_dict:
            return []
        
        cls_preds = output_dict['cls_preds']  # (B, 2, H, W)
        reg_preds = output_dict['reg_preds']  # (B, 14, H, W)
        
        # 从缓存中获取anchor_box
        if not hasattr(self, '_anchor_box_cache'):
            return []
        
        anchor_box = self._anchor_box_cache
        
        # 应用post_process逻辑（简化版本）
        # Step 1: 分类概率
        prob = torch.sigmoid(cls_preds.permute(0, 2, 3, 1))  # (B, H, W, 2)
        prob = prob.reshape(1, -1)  # (1, H*W*2)
        
        # Step 2: 回归预测转换为BBox
        batch_box3d = self._delta_to_boxes3d(reg_preds, anchor_box)  # (B, H*W*2, 7)
        
        # Step 3: 应用score阈值过滤
        score_threshold = 0.3  # 可调参数
        mask = torch.gt(prob, score_threshold).view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        
        # 提取通过阈值的BBox
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        
        if len(boxes3d) == 0:
            return []
        
        results = []
        # 新增：记录当前帧的前景掩码
        frame_fg_mask = np.zeros((self.H, self.W), dtype=bool)
        # 对每个预测的BBox提取特征
        for obj_id, box in enumerate(boxes3d):
            # 解析BBox: (x, y, z, l, w, h, yaw)
            cx, cy, cz = box[0].item(), box[1].item(), box[2].item()
            length, width, height = box[3].item(), box[4].item(), box[5].item()
            yaw = box[6].item()
            
            # 转到像素坐标
            px_center = (cx - self.cav_range[0]) / (self.cav_range[3] - self.cav_range[0]) * self.W
            py_center = (cy - self.cav_range[1]) / (self.cav_range[4] - self.cav_range[1]) * self.H
            
            # 计算旋转后的bbox像素大小
            px_length = (length / (self.cav_range[3] - self.cav_range[0])) * self.W
            px_width = (width / (self.cav_range[4] - self.cav_range[1])) * self.H
            
            # 获取旋转框的四个顶点（考虑yaw角）
            vertices = self._get_rotated_bbox_vertices(px_center, py_center, px_length, px_width, yaw)
            
            # 获取bbox包围框
            x_min = max(0, int(np.floor(np.min(vertices[:, 0]))))
            x_max = min(self.W, int(np.ceil(np.max(vertices[:, 0]))) + 1)
            y_min = max(0, int(np.floor(np.min(vertices[:, 1]))))
            y_max = min(self.H, int(np.ceil(np.max(vertices[:, 1]))) + 1)
            
            if x_max > x_min and y_max > y_min:
                frame_fg_mask[y_min:y_max, x_min:x_max] = True
                # 提取ROI特征（保存所有像素，不取均值）
                roi_feat = fused_feature[0, :, y_min:y_max, x_min:x_max]  # (C, h, w)
                
                # 将特征转为 (num_pixels, C) 形状
                C = roi_feat.shape[0]
                feat_spatial = roi_feat.cpu().detach().numpy()  # (C, h, w)
                feat_spatial = np.transpose(feat_spatial, (1, 2, 0))  # (h, w, C)
                num_pixels = feat_spatial.shape[0] * feat_spatial.shape[1]
                feat_array = feat_spatial.reshape(-1, C)  # (num_pixels, C)
                
                size_ratio = (length / (self.cav_range[3] - self.cav_range[0])) * \
                           (width / (self.cav_range[4] - self.cav_range[1]))
                
                results.append({
                    'features': feat_array,  # (num_pixels, C) - 所有像素的特征
                    'object_id': obj_id,
                    'center': (px_center, py_center),
                    'size': (px_length, px_width),
                    'yaw': float(yaw),
                    'size_ratio': float(size_ratio),
                    'num_pixels': num_pixels,
                    'conf_score': float(scores[obj_id].item()),
                    'method': 'pred_bbox'
                })
        # ================= 新增：背景采样逻辑 =================
        bg_y, bg_x = np.where(~frame_fg_mask)
        total_fg_pixels = sum([r['num_pixels'] for r in results])
        
        # 仅当存在前景且有背景空间时进行采样 (保持1:1或1:2的比例，避免背景淹没前景)
        if len(bg_y) > 0 and total_fg_pixels > 0:
            num_bg_samples = min(len(bg_y), total_fg_pixels * 1) # 1倍背景采样
            sample_indices = np.random.choice(len(bg_y), num_bg_samples, replace=False)
            sampled_bg_y = bg_y[sample_indices]
            sampled_bg_x = bg_x[sample_indices]
            
            # 提取采样的背景特征
            bg_feat = fused_feature[0, :, sampled_bg_y, sampled_bg_x] # (C, num_bg_samples)
            bg_feat_array = bg_feat.T.cpu().detach().numpy() # (num_bg_samples, C)
            
            results.append({
                'features': bg_feat_array,
                'object_id': -1, # 使用 -1 作为背景的特殊ID
                'center': (0, 0),
                'size': (0, 0),
                'yaw': 0.0,
                'size_ratio': 0.0,
                'num_pixels': num_bg_samples,
                'method': 'gt_bbox',
                'is_background': True
            })
        
        return results
    
    def _delta_to_boxes3d(self, deltas: torch.Tensor, anchors: np.ndarray) -> torch.Tensor:
        """
        将回归delta转换为3D BBox
        
        Parameters
        ----------
        deltas : torch.Tensor
            (B, 14, H, W) 回归预测
        anchors : np.ndarray
            (H, W, 2, 7) anchor boxes
        
        Returns
        -------
        boxes3d : torch.Tensor
            (B, H*W*2, 7)
        """
        B = deltas.shape[0]
        deltas_reshaped = deltas.permute(0, 2, 3, 1).contiguous().view(B, -1, 7)
        boxes3d = torch.zeros_like(deltas_reshaped)
        
        if deltas.is_cuda:
            anchors_torch = torch.from_numpy(anchors).float().cuda()
        else:
            anchors_torch = torch.from_numpy(anchors).float()
        
        # (H*W*2, 7)
        anchors_reshaped = anchors_torch.view(-1, 7).float()
        # 计算anchor的对角线
        anchors_d = torch.sqrt(anchors_reshaped[:, 3] ** 2 + anchors_reshaped[:, 4] ** 2)
        anchors_d = anchors_d.repeat(B, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(B, 1, 1)
        
        # 反归一化 xy
        boxes3d[..., [0, 1]] = torch.mul(deltas_reshaped[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        # z
        boxes3d[..., [2]] = torch.mul(deltas_reshaped[..., [2]], anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # lwh (exponential)
        boxes3d[..., [3, 4, 5]] = torch.exp(deltas_reshaped[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw
        boxes3d[..., 6] = deltas_reshaped[..., 6] + anchors_reshaped[..., 6]
        
        return boxes3d
    
    def analyze_dataset(self, num_samples: int = None, extract_method: str = 'gt_bbox'):
        """分析数据集
        
        Args:
            num_samples: 最大处理样本数
            extract_method: 'gt_bbox' - 使用GT bbox（GT标注，精确）
                          'pos_mask' - 使用occ掩码（已弃用，不推荐）
                          'pred_bbox' - 使用模型预测的BBox（推荐！直接体现模型决策）
        """
        print(f"\n{'='*80}")
        print(f"[分析] 遍历数据集 (提取方法: {extract_method})")
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
        total_objects = 0
        total_bg_points = 0
        
        for i, batch_data in enumerate(tqdm(dataloader)):
            try:
                # 前向推理
                with torch.no_grad():
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
                    self.occ_map_list_cache = None
                    self.model_output_cache = None
                    
                    # 前向传播，捕获模型输出
                    model_output = self.model(ego_data)
                    
                    # 缓存模型输出（用于pred_bbox提取）
                    if isinstance(model_output, dict):
                        self.model_output_cache = model_output
                    
                    # 缓存anchor_box（用于pred_bbox转换）
                    if 'anchor_box' in ego_data:
                        self._anchor_box_cache = ego_data['anchor_box'].cpu().numpy()
                    elif 'anchor_box' in batch_data.get('ego', {}):
                        self._anchor_box_cache = batch_data['ego']['anchor_box']
                
                # 获取融合特征
                if self.fused_feature_cache is None:
                    continue
                
                # 提取每个object的特征
                objects = self.extract_object_features(batch_data, self.fused_feature_cache, method=extract_method)
                
                for obj_info in objects:
                    obj_info['sample_id'] = i  # 记录所属样本
                    all_features.append(obj_info)
                    total_objects += 1

                # 随机采样背景特征并加入共同可视化
                if self.enable_background:
                    bg_info = self._extract_random_background_features(self.fused_feature_cache, objects)
                    if bg_info is not None:
                        bg_info['sample_id'] = i
                        all_features.append(bg_info)
                        total_bg_points += bg_info['num_pixels']
            
            except Exception as e:
                print(f"  ⚠ 样本{i}: {str(e)[:80]}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print("✓ 特征提取完成")
        print(f"{'='*80}")
        print(f"处理样本数: {len(dataloader)}")
        print(f"总检测目标数: {total_objects}")
        print(f"每样本平均目标数: {total_objects / len(dataloader):.2f}")
        if self.enable_background:
            print(f"背景随机采样点总数: {total_bg_points}")
            print(f"每样本背景采样点: {self.bg_points_per_sample}")
        print(f"提取方法: {extract_method}")
        
        return all_features
    
    def visualize_object_features(self, num_samples: int, all_features: list, extract_method: str):
        """
        可视化object特征分布
        
        策略：
        1. 用不同颜色表示不同object
        2. 降维到2D (UMAP或PCA)
        3. 展示所有objects的特征分布（展平所有像素）
        
        核心改变：
        - 之前：每个object = 1个均值特征向量
        - 现在：每个object = N个像素特征向量（保留空间信息）
        - 可视化时：每个像素点代表一个空间位置，同一object用同一颜色
        """
        if len(all_features) == 0:
            print("❌ 无特征可视化")
            return
        print(f"\n{'='*80}")
        print("[可视化] 准备特征可视化...")
        print(f"{'='*80}\n")
        
        # 展平所有特征：展开每个entry中的特征
        all_features_flat = []
        object_ids_flat = []
        sample_ids_flat = []
        spatial_positions_flat = []  # 新增：记录空间位置

        print(f"特征聚合模式: {self.object_feature_mode}")
        
        for f in all_features:
            if 'features' in f:
                feat_pixels = f['features']  # (num_pixels, C)
                if self.object_feature_mode == 'mean':
                    feat_mean = feat_pixels.mean(axis=0, keepdims=True)  # (1, C)
                    all_features_flat.append(feat_mean)
                    object_ids_flat.append(f['object_id'])
                    sample_ids_flat.append(f['sample_id'])

                    if f.get('object_id') == -1:
                        spatial_pos = np.array([[0.5, 0.5]], dtype=np.float32)
                    else:
                        cx, cy = f['center']
                        spatial_pos = np.array([[cx / self.W, cy / self.H]], dtype=np.float32)
                        spatial_pos = np.clip(spatial_pos, 0, 1)

                    spatial_positions_flat.append(spatial_pos)
                else:
                    num_pixels = feat_pixels.shape[0]
                    all_features_flat.append(feat_pixels)
                    object_ids_flat.extend([f['object_id']] * num_pixels)
                    sample_ids_flat.extend([f['sample_id']] * num_pixels)

                    # 新增：为背景特征添加浓缩的空间位置编码
                    if f.get('object_id') == -1:
                        # 背景点：随机分布在整个空间中
                        spatial_pos = np.random.rand(num_pixels, 2)
                    else:
                        # 前景点：聚集在bbox中心附近
                        cx, cy = f['center']
                        sx, sy = f['size']
                        # 生成高斯分布的位置编码
                        spatial_pos = np.random.normal(loc=[cx/self.W, cy/self.H],
                                                       scale=[sx/self.W*0.3, sy/self.H*0.3],
                                                       size=(num_pixels, 2))
                        spatial_pos = np.clip(spatial_pos, 0, 1)

                    spatial_positions_flat.append(spatial_pos)
                
            elif 'feature' in f:
                feat_vec = f['feature'].reshape(1, -1)
                all_features_flat.append(feat_vec)
                object_ids_flat.append(f['object_id'])
                sample_ids_flat.append(f['sample_id'])
                if f.get('object_id', -1) == -1:
                    spatial_positions_flat.append(np.array([[0.5, 0.5]], dtype=np.float32))
                else:
                    cx, cy = f.get('center', (self.W / 2.0, self.H / 2.0))
                    pos = np.array([[cx / self.W, cy / self.H]], dtype=np.float32)
                    spatial_positions_flat.append(np.clip(pos, 0, 1))
        
        X = np.vstack(all_features_flat)  # (total_pixels, C)
        spatial_pos = np.vstack(spatial_positions_flat)  # (total_pixels, 2)
        object_ids = np.array(object_ids_flat)  # (total_pixels,)
        sample_ids = np.array(sample_ids_flat)  # (total_pixels,)
        
        # 标准化
        
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"不同object数: {len(np.unique(object_ids))}")
        
        # 特征标准化
        print("\n标准化特征...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 新增：将空间位置信息融合到特征中
        print("融合空间位置信息...")
        spatial_weight = 0.2  # 空间权重：占20%（提高特征的类别分离性）
        X_with_spatial = np.hstack([
            X_scaled * (1 - spatial_weight),  # 原始特征：80%
            spatial_pos * spatial_weight      # 空间位置：20%
        ])
        
        # 降维
        print("降维...")
        if UMAP_AVAILABLE:
            print("  使用 UMAP (3D，增强类别分离)...")
            # 增强参数：更小的min_dist + 更多n_neighbors 有助于类别分离
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.01, n_components=3, 
                               metric='euclidean', random_state=42)
            X_embed = reducer.fit_transform(X_with_spatial)
            method = "UMAP"
            is_3d = True
        else:
            print("  使用 PCA (UMAP不可用, 2D)...")
            pca = PCA(n_components=2, random_state=42)
            X_embed = pca.fit_transform(X_scaled)
            method = "PCA"
            is_3d = False

        if self.interactive_3d and is_3d:
            if not PLOTLY_AVAILABLE:
                print("⚠ 交互式3D需要plotly，当前环境未安装，回退到静态图。")
            else:
                print("生成交互式3D可视化 (Plotly HTML)...")
                bg_mask = object_ids == -1
                fg_object_ids = np.unique(object_ids[object_ids != -1])
                fg_sample_ids = np.unique(sample_ids[~bg_mask])

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=(
                        f'BEV Features by Object ({method})',
                        f'BEV Features by Sample ({method})'
                    )
                )

                if np.any(bg_mask):
                    bg_kwargs = dict(
                        mode='markers',
                        marker=dict(size=2, color='black', opacity=0.25),
                        name='Background',
                        legendgroup='bg',
                        showlegend=True
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=X_embed[bg_mask, 0],
                            y=X_embed[bg_mask, 1],
                            z=X_embed[bg_mask, 2],
                            **bg_kwargs
                        ),
                        row=1,
                        col=1
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=X_embed[bg_mask, 0],
                            y=X_embed[bg_mask, 1],
                            z=X_embed[bg_mask, 2],
                            mode='markers',
                            marker=dict(size=2, color='black', opacity=0.25),
                            name='Background (All Samples)',
                            legendgroup='bg',
                            showlegend=False
                        ),
                        row=1,
                        col=2
                    )

                if self.vis_color_mode == 'fg_bg_two_color':
                    fg_mask = ~bg_mask
                    if np.any(fg_mask):
                        fig.add_trace(
                            go.Scatter3d(
                                x=X_embed[fg_mask, 0],
                                y=X_embed[fg_mask, 1],
                                z=X_embed[fg_mask, 2],
                                mode='markers',
                                marker=dict(size=3, color='#1f77b4', opacity=0.75),
                                name='Object(Foreground)',
                                legendgroup='fg',
                                showlegend=True
                            ),
                            row=1,
                            col=1
                        )
                else:
                    obj_colors = plt.cm.tab20(np.linspace(0, 1, min(len(fg_object_ids), 20)))
                    if len(fg_object_ids) > 20:
                        obj_colors = np.vstack([obj_colors] * (len(fg_object_ids) // 20 + 1))[:len(fg_object_ids)]

                    for color_idx, obj_id in enumerate(fg_object_ids):
                        mask = object_ids == obj_id
                        color_rgba = obj_colors[color_idx % len(obj_colors)]
                        color_hex = '#{:02x}{:02x}{:02x}'.format(
                            int(color_rgba[0] * 255), int(color_rgba[1] * 255), int(color_rgba[2] * 255)
                        )
                        fig.add_trace(
                            go.Scatter3d(
                                x=X_embed[mask, 0],
                                y=X_embed[mask, 1],
                                z=X_embed[mask, 2],
                                mode='markers',
                                marker=dict(size=3, color=color_hex, opacity=0.75),
                                name=f'Object {obj_id}',
                                legendgroup=f'obj_{obj_id}',
                                showlegend=True
                            ),
                            row=1,
                            col=1
                        )

                sample_colors = plt.cm.hsv(np.linspace(0, 1, min(len(fg_sample_ids), 20)))
                if len(fg_sample_ids) > 20:
                    sample_colors = np.vstack([sample_colors] * (len(fg_sample_ids) // 20 + 1))[:len(fg_sample_ids)]

                for color_idx, sample_id in enumerate(fg_sample_ids):
                    mask = (sample_ids == sample_id) & (~bg_mask)
                    color_rgba = sample_colors[color_idx % len(sample_colors)]
                    color_hex = '#{:02x}{:02x}{:02x}'.format(
                        int(color_rgba[0] * 255), int(color_rgba[1] * 255), int(color_rgba[2] * 255)
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=X_embed[mask, 0],
                            y=X_embed[mask, 1],
                            z=X_embed[mask, 2],
                            mode='markers',
                            marker=dict(size=3, color=color_hex, opacity=0.75),
                            name=f'Sample {sample_id}',
                            legendgroup=f'sample_{sample_id}',
                            showlegend=True
                        ),
                        row=1,
                        col=2
                    )

                fig.update_layout(
                    width=1600,
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=50),
                    scene=dict(xaxis_title=f'{method} 1', yaxis_title=f'{method} 2', zaxis_title=f'{method} 3'),
                    scene2=dict(xaxis_title=f'{method} 1', yaxis_title=f'{method} 2', zaxis_title=f'{method} 3')
                )

                dim_tag = '3d'
                html_path = self.output_dir / f'object_features_{dim_tag}_{method.lower()}_{extract_method}_samples{num_samples}_interactive.html'
                fig.write_html(str(html_path), include_plotlyjs='cdn')
                print(f"✓ 交互式可视化已保存: {html_path}")
                return X_embed, object_ids, sample_ids
        
        # 可视化
        if is_3d:
            fig = plt.figure(figsize=(20, 8))
            ax_obj = fig.add_subplot(1, 2, 1, projection='3d')
            ax_sample = fig.add_subplot(1, 2, 2, projection='3d')
            axes = [ax_obj, ax_sample]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 左图：按object着色
        print("绘制左图 (按object着色)...")
        ax = axes[0]
        # --- 分离背景与前景 ---
        bg_mask = object_ids == -1
        fg_object_ids = np.unique(object_ids[object_ids != -1])
        n_objects = len(fg_object_ids)
        if self.vis_color_mode == 'fg_bg_two_color':
            fg_mask = ~bg_mask
            if np.any(bg_mask):
                if is_3d:
                    ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1], X_embed[bg_mask, 2],
                               c='black',
                               label='Background',
                               s=16, alpha=0.25, edgecolors='none', zorder=1)
                else:
                    ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1],
                               c='black',
                               label='Background',
                               s=16, alpha=0.25, edgecolors='none', zorder=1)
            if np.any(fg_mask):
                if is_3d:
                    ax.scatter(X_embed[fg_mask, 0], X_embed[fg_mask, 1], X_embed[fg_mask, 2],
                               c='#1f77b4',
                               label='Object(Foreground)',
                               s=30, alpha=0.7, edgecolors='black', linewidth=0.3, zorder=2)
                else:
                    ax.scatter(X_embed[fg_mask, 0], X_embed[fg_mask, 1],
                               c='#1f77b4',
                               label='Object(Foreground)',
                               s=36, alpha=0.7, edgecolors='black', linewidth=0.35, zorder=2)
        else:
            # 1. 首先绘制背景特征（特殊颜色）
            if np.any(bg_mask):
                if is_3d:
                    ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1], X_embed[bg_mask, 2],
                               c='black',
                               label="Background",
                               s=16, alpha=0.25, edgecolors='none', zorder=1)
                else:
                    ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1],
                               c='black',
                               label="Background",
                               s=16, alpha=0.25, edgecolors='none', zorder=1)
            # 为每个object分配不同的颜色
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_objects, 20)))
            if n_objects > 20:
                colors = np.vstack([colors] * (n_objects // 20 + 1))[:n_objects]
            
            for color_idx, obj_id in enumerate(fg_object_ids):
                mask = object_ids == obj_id
                if is_3d:
                    ax.scatter(X_embed[mask, 0], X_embed[mask, 1], X_embed[mask, 2],
                               c=[colors[color_idx % len(colors)]], 
                               label=f"Object {obj_id}",
                               s=30, alpha=0.7, edgecolors='black', linewidth=0.3, zorder=2)
                else:
                    ax.scatter(X_embed[mask, 0], X_embed[mask, 1],
                               c=[colors[color_idx % len(colors)]], 
                               label=f"Object {obj_id}",
                               s=36, alpha=0.7, edgecolors='black', linewidth=0.35, zorder=2)
        
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
        if is_3d:
            ax.set_zlabel(f'{method} 3')
        if self.vis_color_mode == 'fg_bg_two_color':
            ax.set_title(f'BEV Features by Foreground vs Background ({method})')
        else:
            ax.set_title(f'BEV Features by Object ({method})')
        ax.grid(True, alpha=0.3)
        if n_objects <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 右图：按sample着色
        print("绘制右图 (按sample着色)...")
        ax = axes[1]
        # 同样先画背景（特殊颜色）
        if np.any(bg_mask):
            if is_3d:
                ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1], X_embed[bg_mask, 2],
                           c='black',
                           label="Background (All Samples)",
                           s=16, alpha=0.25, edgecolors='none', zorder=1)
            else:
                ax.scatter(X_embed[bg_mask, 0], X_embed[bg_mask, 1],
                           c='black',
                           label="Background (All Samples)",
                           s=16, alpha=0.25, edgecolors='none', zorder=1)
        fg_sample_ids = np.unique(sample_ids[~bg_mask])
        n_samples = len(fg_sample_ids)
        colors2 = plt.cm.hsv(np.linspace(0, 1, min(n_samples, 20)))
        if n_samples > 20:
            colors2 = np.vstack([colors2] * (n_samples // 20 + 1))[:n_samples]
        
        for color_idx, sample_id in enumerate(fg_sample_ids):
            # 仅对前景object点按sample着色，避免覆盖背景黑色点
            mask = (sample_ids == sample_id) & (~bg_mask)
            if is_3d:
                ax.scatter(X_embed[mask, 0], X_embed[mask, 1], X_embed[mask, 2],
                           c=[colors2[color_idx % len(colors2)]],
                           label=f"Sample {sample_id}",
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.3, zorder=2)
            else:
                ax.scatter(X_embed[mask, 0], X_embed[mask, 1],
                           c=[colors2[color_idx % len(colors2)]],
                           label=f"Sample {sample_id}",
                           s=36, alpha=0.7, edgecolors='black', linewidth=0.35, zorder=2)
        
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
        if is_3d:
            ax.set_zlabel(f'{method} 3')
        ax.set_title(f'BEV Features by Sample ({method})')
        ax.grid(True, alpha=0.3)
        if n_samples <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        dim_tag = '3d' if is_3d else '2d'
        output_path = self.output_dir / f'object_features_{dim_tag}_{method.lower()}_{extract_method}_samples{num_samples}.png'
        print(f"✓ 保存: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return X_embed, object_ids, sample_ids
    
    def generate_statistics(self, all_features: list):
        """生成统计信息"""
        print(f"\n{'='*80}")
        print("[统计] 特征分布统计")
        print(f"{'='*80}\n")
        
        # 按object统计
        object_stats = defaultdict(list)
        for f in all_features:
            oid = f['object_id']
            object_stats[oid].append(f)

        background_entries = object_stats.pop(-1, []) if -1 in object_stats else []
        
        # 计算总像素数（所有object的所有像素）
        total_pixel_features = sum(f['features'].shape[0] for f in all_features)
        
        stats_data = {
            'total_objects': len(object_stats),
            'total_pixel_features': total_pixel_features,  # 新增：所有像素点的特征数
            'total_extraction_entries': len(all_features),  # object提取条目数
            'background_entries': len(background_entries),
            'background_pixel_features': int(sum(f['features'].shape[0] for f in background_entries)) if background_entries else 0,
            'object_info': {}
        }
        
        # === 诊断信息 ===
        print("[诊断] 背景/前景特征分离度分析...")
        if len(background_entries) > 0 and len(object_stats) > 0:
            # 合并所有背景特征
            bg_features = np.vstack([f['features'] for f in background_entries])
            
            # 合并所有前景特征
            fg_features = np.vstack([f['features'] for f in sum(object_stats.values(), [])])
            
            # 计算统计指标
            bg_mean = bg_features.mean(axis=0)
            fg_mean = fg_features.mean(axis=0)
            
            # 距离指标
            mean_dist = np.linalg.norm(bg_mean - fg_mean)
            
            # 方差分析
            bg_var = bg_features.var(axis=0).mean()
            fg_var = fg_features.var(axis=0).mean()
            
            print(f"  背景样本数: {len(bg_features)}")
            print(f"  前景样本数: {len(fg_features)}")
            print(f"  特征维度: {bg_features.shape[1]}")
            print(f"  背景均值-前景均值距离: {mean_dist:.4f}")
            print(f"  背景特征方差: {bg_var:.4f}")
            print(f"  前景特征方差: {fg_var:.4f}")
            
            if mean_dist < 0.5:
                print(f"\n  ⚠️ 背景/前景特征在原始空间中距离很小！")
                print(f"     ✓ 好消息：已添加空间位置信息，应有改善")
                print(f"     💡 如果仍不理想，可尝试：")
                print(f"        - --enable_background: 确保启用背景采样")
                print(f"        - 增加 --max_samples 获得更多样本数据")
                print(f"        - 检查 BEV 特征提取层是否区分了前景/背景")
            elif mean_dist < 1.5:
                print(f"  ✓ 背景/前景特征距离中等，降维后应有明显分离")
            else:
                print(f"  ✓✓ 背景/前景特征距离很大，应有明显的类别分离")
        
        for oid, features in object_stats.items():
            # 计算该object的总像素数
            total_pixels_in_obj = sum(f['features'].shape[0] for f in features)
            avg_pixels_per_entry = total_pixels_in_obj / len(features)
            
            stats_data['object_info'][str(oid)] = {
                'num_extraction_entries': len(features),  # 该object被提取的次数（每个样本一次）
                'total_pixel_features': total_pixels_in_obj,  # 该object的总像素数
                'avg_pixels_per_entry': float(avg_pixels_per_entry),
                'avg_size': float(np.mean([f['size_ratio'] for f in features])),
                'avg_bbox_pixels': float(np.mean([f['num_pixels'] for f in features]))
            }
            print(f"  Object {oid}: {len(features)} 次提取, "
                  f"总像素特征数: {total_pixels_in_obj}, "
                  f"每次提取平均像素: {avg_pixels_per_entry:.1f}")
        
        # 保存统计
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"\n✓ 统计信息已保存: {stats_path}")
        
        return stats_data
    
    def run(self, num_samples: int = None, extract_method: str = 'gt_bbox'):
        """运行完整分析"""
        # 提取特征
        all_features = self.analyze_dataset(num_samples=num_samples, extract_method=extract_method)
        
        if len(all_features) == 0:
            print("❌ 未提取到特征")
            return
        
        # 可视化
        self.visualize_object_features(num_samples,all_features, extract_method)
        
        # 统计
        stats = self.generate_statistics(all_features)
        
        print(f"\n{'='*80}")
        print(f"✓ 分析完成")
        print(f"  结果保存到: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='BEV Feature Distribution Analysis v3')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config yaml')
    parser.add_argument('--output', default='./bev_analysis_results', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--extract_method', default='gt_bbox', choices=['gt_bbox', 'pos_mask', 'pred_bbox'],
                       help='Method to extract ROI: gt_bbox (GT bbox with yaw) | pos_mask (deprecated) | pred_bbox (model predictions, recommended!)')
    parser.add_argument('--enable_background', action='store_true',
                       help='Enable random background feature sampling for joint visualization')
    parser.add_argument('--bg_points_per_sample', type=int, default=1000,
                       help='Number of random background points sampled per sample')
    parser.add_argument('--bg_seed', type=int, default=42,
                       help='Random seed for background sampling')
    parser.add_argument('--interactive_3d', action='store_true',
                       help='Export interactive 3D HTML visualization (requires plotly, UMAP 3D only)')
    parser.add_argument('--object_feature_mode', default='all_pixels', choices=['all_pixels', 'mean'],
                       help='Feature aggregation in each object region: all_pixels | mean')
    parser.add_argument('--vis_color_mode', default='multi_object', choices=['multi_object', 'fg_bg_two_color'],
                       help='Color mode for object plot: multi_object (one color per object) | fg_bg_two_color (foreground/background two colors)')
    
    args = parser.parse_args()
    
    analyzer = BEVFeatureAnalyzerV3(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        output_dir=args.output,
        enable_background=args.enable_background,
        bg_points_per_sample=args.bg_points_per_sample,
        bg_seed=args.bg_seed,
        interactive_3d=args.interactive_3d,
        object_feature_mode=args.object_feature_mode,
        vis_color_mode=args.vis_color_mode,
    )
    
    analyzer.run(num_samples=args.max_samples, extract_method=args.extract_method)


if __name__ == '__main__':
    main()
