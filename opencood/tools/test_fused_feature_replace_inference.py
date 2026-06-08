#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# python opencood/tools/test_fused_feature_replace_inference.py 
#     --model /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/net_epoch37.pth 
#     --config /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/config.yaml 
#     --output /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/feature_replace_test_rotation 
#     --fusion_method intermediate --sample_idx 0 --score_threshold 0.5
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Subset

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.visualization import simple_vis


class FusedFeatureRegionReplaceTester:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        output_dir: str,
        fusion_method: str = 'intermediate',
        device: str = 'cuda',
        sample_idx: int = 0,
        source_sample_idx: int = None,
        score_threshold: float = 0.5,
        max_transforms: int = 6,
    ):
        self.model_path = model_path
        self.target_sample_idx = int(sample_idx)
        self.source_sample_idx = int(source_sample_idx) if source_sample_idx is not None else int(sample_idx)
        self.sample_idx = self.target_sample_idx
        self.config_path = config_path
        self.output_dir = Path(output_dir) / f'{self.source_sample_idx}_{self.target_sample_idx}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fusion_method = fusion_method
        
        self.score_threshold = score_threshold
        self.max_transforms = max(1, int(max_transforms))

        self.device = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')

        self.hypes = yaml_utils.load_yaml(config_path)
        self.pc_range = self.hypes['postprocess']['gt_range']
        self.cav_range = self.hypes.get('cav_lidar_range', [-102.4, -102.4, -3, 102.4, 102.4, 1])

        self.model = train_utils.create_model(self.hypes)
        self.model.to(self.device)
        self.model.eval()
        self._load_checkpoint(model_path)

        self.dataset = build_dataset(self.hypes, visualize=True, train=False)

        self.left_hand = True if (
            ('OPV2V' in self.hypes['test_dir']) or ('V2XSET' in self.hypes['test_dir'])
        ) else False

    def _load_checkpoint(self, checkpoint_path: str):
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

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict, strict=False)

    def _get_batch_by_idx(self, sample_idx: int):
        subset = Subset(self.dataset, [int(sample_idx)])
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_batch_test,
            pin_memory=False,
            drop_last=False,
        )
        return next(iter(loader))

    def _run_inference_once(self, batch_data_cpu: dict, inject_feature: torch.Tensor = None):
        captured = {'fused_feature': None}

        def capture_fused_feature(module, inputs, outputs):
            if isinstance(inputs, tuple) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                captured['fused_feature'] = inputs[0].detach().clone()

        def replace_fused_feature(module, inputs):
            if inject_feature is None:
                return None
            x = inputs[0]
            replace = inject_feature.to(device=x.device, dtype=x.dtype)
            return (replace,)

        hook_capture = self.model.cls_head.register_forward_hook(capture_fused_feature)
        hook_replace_list = []
        if inject_feature is not None:
            # 在所有三个检测头上都注册 pre-hook，确保它们都使用替换后的feature
            hook_replace_list.append(self.model.cls_head.register_forward_pre_hook(replace_fused_feature))
            hook_replace_list.append(self.model.reg_head.register_forward_pre_hook(replace_fused_feature))
            hook_replace_list.append(self.model.dir_head.register_forward_pre_hook(replace_fused_feature))

        with torch.no_grad():
            batch_data = train_utils.to_device(copy.deepcopy(batch_data_cpu), self.device)

            if self.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data, self.model, self.dataset)
            elif self.fusion_method == 'early':
                infer_result, _ = inference_utils.inference_early_fusion(batch_data, self.model, self.dataset)
            elif self.fusion_method == 'intermediate':
                infer_result, _ = inference_utils.inference_intermediate_fusion(batch_data, self.model, self.dataset)
            elif self.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data, self.model, self.dataset)
            elif self.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data, self.model, self.dataset)
            elif self.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data, self.model, self.dataset, single_gt=True)
            else:
                raise ValueError(f'Unsupported fusion_method: {self.fusion_method}')

        hook_capture.remove()
        if hook_replace_list:
            for hook in hook_replace_list:
                hook.remove()

        if captured['fused_feature'] is None:
            raise RuntimeError('Failed to capture fused_feature from cls_head input.')

        return infer_result, captured['fused_feature']

    def _world_to_feature_xy(self, x_world: np.ndarray, y_world: np.ndarray, feat_h: int, feat_w: int):
        x_min, y_min, _, x_max, y_max, _ = self.cav_range
        px = (x_world - x_min) / (x_max - x_min) * feat_w
        py = (y_world - y_min) / (y_max - y_min) * feat_h
        return px, py

    def _bbox_corners_to_rect(self, box_corners: np.ndarray, feat_h: int, feat_w: int):
        x_world = box_corners[:, 0]
        y_world = box_corners[:, 1]
        px, py = self._world_to_feature_xy(x_world, y_world, feat_h, feat_w)

        x0 = int(np.floor(np.min(px)))
        x1 = int(np.ceil(np.max(px))) + 1
        y0 = int(np.floor(np.min(py)))
        y1 = int(np.ceil(np.max(py))) + 1

        x0 = max(0, min(x0, feat_w - 1))
        x1 = max(1, min(x1, feat_w))
        y0 = max(0, min(y0, feat_h - 1))
        y1 = max(1, min(y1, feat_h))

        if x1 <= x0 or y1 <= y0:
            return None
        return (y0, y1, x0, x1)

    def _select_source_rect(self, infer_result: dict, feat_h: int, feat_w: int):
        pred_box_tensor = infer_result.get('pred_box_tensor', None)
        pred_score = infer_result.get('pred_score', None)

        if pred_box_tensor is None or len(pred_box_tensor) == 0:
            return None, None

        boxes = pred_box_tensor.detach().cpu().numpy()
        if pred_score is not None and len(pred_score) == len(pred_box_tensor):
            scores = pred_score.detach().cpu().numpy()
            candidate_idx = np.argsort(-scores)
        else:
            candidate_idx = np.arange(len(boxes))

        for idx in candidate_idx:
            if pred_score is not None and pred_score[idx].item() < self.score_threshold:
                continue
            rect = self._bbox_corners_to_rect(boxes[idx], feat_h, feat_w)
            if rect is None:
                continue
            y0, y1, x0, x1 = rect
            if (y1 - y0) * (x1 - x0) >= 9:
                return rect, int(idx)

        # fallback: choose the first valid prediction
        for idx in candidate_idx:
            rect = self._bbox_corners_to_rect(boxes[idx], feat_h, feat_w)
            if rect is not None:
                return rect, int(idx)

        return None, None

    @staticmethod
    def _rect_overlap(rect_a, rect_b):
        ay0, ay1, ax0, ax1 = rect_a
        by0, by1, bx0, bx1 = rect_b
        overlap_y = not (ay1 <= by0 or ay0 >= by1)
        overlap_x = not (ax1 <= bx0 or ax0 >= bx1)
        return overlap_y and overlap_x

    @staticmethod
    def _fit_rect_at_anchor(anchor_y, anchor_x, patch_h, patch_w, feat_h, feat_w):
        ty0 = int(max(0, min(anchor_y, feat_h - patch_h)))
        tx0 = int(max(0, min(anchor_x, feat_w - patch_w)))
        ty1 = ty0 + patch_h
        tx1 = tx0 + patch_w
        if patch_h <= 0 or patch_w <= 0 or ty1 > feat_h or tx1 > feat_w:
            return None
        return (ty0, ty1, tx0, tx1)

    def _choose_target_rects(self, source_rect, feat_h: int, feat_w: int):
        sy0, sy1, sx0, sx1 = source_rect
        src_h = sy1 - sy0
        src_w = sx1 - sx0

        transform_specs = [
            {'name': 'translate', 'op': 'identity', 'scale': 1.0},
            {'name': 'rotate90', 'op': 'rot90', 'scale': 1.0},
            {'name': 'rotate180', 'op': 'rot180', 'scale': 1.0},
            {'name': 'rotate270', 'op': 'rot270', 'scale': 1.0},
            {'name': 'flip_horizontal', 'op': 'flip_h', 'scale': 1.0},
            {'name': 'scale2x', 'op': 'identity', 'scale': 2.0},
            {'name': 'scale2x_rotate90', 'op': 'rot90', 'scale': 2.0},
        ]

        anchor_candidates = [
            (2, 2),
            (2, feat_w // 2),
            (2, max(2, feat_w - 4)),
            (feat_h // 2, 2),
            (feat_h // 2, max(2, feat_w - 4)),
            (max(2, feat_h - 4), 2),
            (max(2, feat_h - 4), feat_w // 2),
            (max(2, feat_h - 4), max(2, feat_w - 4)),
        ]

        plans = []
        used_target_rects = []

        for spec in transform_specs:
            scale = spec['scale']
            op = spec['op']

            base_h = src_h
            base_w = src_w
            if op in ('rot90', 'rot270'):
                base_h, base_w = src_w, src_h

            out_h = int(max(1, min(feat_h, round(base_h * scale))))
            out_w = int(max(1, min(feat_w, round(base_w * scale))))

            selected_target = None
            for ay, ax in anchor_candidates:
                rect = self._fit_rect_at_anchor(ay, ax, out_h, out_w, feat_h, feat_w)
                if rect is None:
                    continue
                if self._rect_overlap(rect, source_rect):
                    continue
                if any(self._rect_overlap(rect, old_rect) for old_rect in used_target_rects):
                    continue
                selected_target = rect
                break

            if selected_target is None:
                fallback = self._fit_rect_at_anchor(2, 2, out_h, out_w, feat_h, feat_w)
                if fallback is None:
                    continue
                selected_target = fallback

            used_target_rects.append(selected_target)
            plans.append({
                'name': spec['name'],
                'op': op,
                'scale': scale,
                'source_rect': source_rect,
                'target_rect': selected_target,
                'target_hw': (out_h, out_w),
            })

            if len(plans) >= self.max_transforms:
                break

        return plans

    @staticmethod
    def _transform_patch(patch: torch.Tensor, op: str):
        """
        Apply geometric transformation (rotation/flip) preserving natural post-transform shape.
        
        Args:
            patch: (B, C, H, W) tensor
            op: 'identity', 'rot90', 'rot180', 'rot270', 'flip_h'
        
        Returns:
            Transformed patch with natural post-transform shape:
            - rot90/rot270: (B, C, W, H)  [dimensions swap]
            - rot180/flip_h/identity: (B, C, H, W)  [shape preserved]
        """
        if op == 'rot90':
            patch = torch.rot90(patch, 1, dims=(-2, -1))  # (B,C,H,W) -> (B,C,W,H)
        elif op == 'rot180':
            patch = torch.rot90(patch, 2, dims=(-2, -1))  # (B,C,H,W) -> (B,C,H,W)
        elif op == 'rot270':
            patch = torch.rot90(patch, 3, dims=(-2, -1))  # (B,C,H,W) -> (B,C,W,H)
        elif op == 'flip_h':
            patch = torch.flip(patch, dims=(-1,))  # (B,C,H,W) -> (B,C,H,W)
        # else: op == 'identity', return as-is
        
        return patch

    def _apply_region_replace(self, fused_feature: torch.Tensor, plan: dict, source_feature: torch.Tensor = None):
        """
        Replace target region with transformed source patch.
        
        Key principle: The shape of the replacement region matches exactly 
        the natural post-transform shape (no distortion via interpolation).
        
        Workflow:
        1. Extract source patch
        2. Apply rotation/flip (preserves natural post-transform shape)
        3. Apply scale if included in plan
        4. Replace target region using the actual transformed patch shape
           without any additional interpolation (ensures geometric consistency)
        """
        patched = fused_feature.clone()
        sy0, sy1, sx0, sx1 = plan['source_rect']
        ty0, tx0 = plan['target_rect'][0], plan['target_rect'][2]  # Start position
        feat_h, feat_w = fused_feature.shape[-2:]
        source_tensor = source_feature if source_feature is not None else fused_feature
        
        # Extract and transform
        source_patch = source_tensor[:, :, sy0:sy1, sx0:sx1]  # (B, C, src_h, src_w)
        transformed_patch = self._transform_patch(source_patch, plan['op'])  # Natural post-transform shape
        
        # Apply scale if present (e.g., scale2x, scale2x_rotate90)
        if plan['scale'] != 1.0:
            scale_h = int(round(transformed_patch.shape[-2] * plan['scale']))
            scale_w = int(round(transformed_patch.shape[-1] * plan['scale']))
            transformed_patch = F.interpolate(
                transformed_patch,
                size=(scale_h, scale_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Get actual transformed shape (WITH rotation/scale effects, NO distortion)
        actual_h, actual_w = transformed_patch.shape[-2:]
        
        # Ensure target region doesn't exceed feature map boundaries
        ty0_clamped = max(0, min(ty0, feat_h - actual_h))
        tx0_clamped = max(0, min(tx0, feat_w - actual_w))
        ty1_clamped = ty0_clamped + actual_h
        tx1_clamped = tx0_clamped + actual_w
        
        # Direct replacement: shape matches exactly, no interpolation
        patched[:, :, ty0_clamped:ty1_clamped, tx0_clamped:tx1_clamped] = transformed_patch
        
        return patched

    @staticmethod
    def _feature_mag_map(feature_tensor: torch.Tensor):
        feat = feature_tensor.detach().cpu().numpy()[0]  # (C, H, W)
        mag = np.linalg.norm(feat, axis=0)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        return mag

    @staticmethod
    def _draw_rect(ax, rect, color, label):
        y0, y1, x0, x1 = rect
        patch = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        ax.text(x0 + 2, y0 + 2, label, color=color, fontsize=9, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1))

    def run(self):
        print('[1/6] 准备source/target样本数据...')
        source_batch_data_cpu = self._get_batch_by_idx(self.source_sample_idx)
        target_batch_data_cpu = self._get_batch_by_idx(self.target_sample_idx)

        print('[2/6] source样本正常推理并提取高置信目标区域...')
        source_result, source_fused = self._run_inference_once(source_batch_data_cpu, inject_feature=None)

        _, _, source_feat_h, source_feat_w = source_fused.shape
        source_rect, source_idx = self._select_source_rect(source_result, source_feat_h, source_feat_w)
        if source_rect is None:
            raise RuntimeError('source样本未找到可用预测框区域，无法执行跨样本特征替换测试。')

        print('[3/6] target样本正常推理并捕获融合特征...')
        target_baseline_result, target_baseline_fused = self._run_inference_once(target_batch_data_cpu, inject_feature=None)

        _, _, target_feat_h, target_feat_w = target_baseline_fused.shape
        transform_plans = self._choose_target_rects(source_rect, target_feat_h, target_feat_w)
        if len(transform_plans) == 0:
            raise RuntimeError('未生成可用的target区域变换方案。')

        print('[4/6] 将source特征替换到target边缘区域并逐个推理...')
        variant_results = []
        for plan in transform_plans:
            patched_fused = self._apply_region_replace(
                target_baseline_fused,
                plan,
                source_feature=source_fused,
            )
            modified_result, modified_fused = self._run_inference_once(target_batch_data_cpu, inject_feature=patched_fused)
            variant_results.append({
                'plan': plan,
                'infer_result': modified_result,
                'fused_feature': modified_fused,
            })

        print('[5/6] 生成source->target替换推理可视化...')
        prefix = f'src{self.source_sample_idx}_tgt{self.target_sample_idx}'
        source_vis_path = self.output_dir / f'{prefix}_source_baseline_inference_bev.png'
        target_base_vis_path = self.output_dir / f'{prefix}_target_baseline_inference_bev.png'
        variant_vis_paths = []

        source_origin_lidar = source_batch_data_cpu['ego']['origin_lidar'][0]
        target_origin_lidar = target_batch_data_cpu['ego']['origin_lidar'][0]

        simple_vis.visualize(
            source_result,
            source_origin_lidar,
            self.pc_range,
            str(source_vis_path),
            method='bev',
            left_hand=self.left_hand,
        )
        simple_vis.visualize(
            target_baseline_result,
            target_origin_lidar,
            self.pc_range,
            str(target_base_vis_path),
            method='bev',
            left_hand=self.left_hand,
        )

        for item in variant_results:
            name = item['plan']['name']
            vis_path = self.output_dir / f'{prefix}_modified_inference_bev_{name}.png'
            simple_vis.visualize(
                item['infer_result'],
                target_origin_lidar,
                self.pc_range,
                str(vis_path),
                method='bev',
                left_hand=self.left_hand,
            )
            variant_vis_paths.append(vis_path)

        source_img = plt.imread(str(source_vis_path))
        target_base_img = plt.imread(str(target_base_vis_path))
        source_mag = self._feature_mag_map(source_fused)
        target_base_mag = self._feature_mag_map(target_baseline_fused)
        n_cols = len(variant_results) + 2
        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))

        axes[0, 0].imshow(source_img)
        axes[0, 0].set_title(f'Source Baseline (id={self.source_sample_idx})')
        axes[0, 0].axis('off')

        ax = axes[1, 0]
        ax.imshow(source_mag, cmap='viridis')
        self._draw_rect(ax, source_rect, 'red', 'Source')
        ax.set_title('Source Fused Feature')
        ax.set_xlabel('W')
        ax.set_ylabel('H')

        axes[0, 1].imshow(target_base_img)
        axes[0, 1].set_title(f'Target Baseline (id={self.target_sample_idx})')
        axes[0, 1].axis('off')

        ax = axes[1, 1]
        ax.imshow(target_base_mag, cmap='viridis')
        ax.set_title('Target Baseline Fused Feature')
        ax.set_xlabel('W')
        ax.set_ylabel('H')

        for idx, item in enumerate(variant_results, start=2):
            plan = item['plan']
            mod_img = plt.imread(str(variant_vis_paths[idx - 2]))
            mod_mag = self._feature_mag_map(item['fused_feature'])

            axes[0, idx].imshow(mod_img)
            axes[0, idx].set_title(f"{plan['name']} | {plan['op']} x{plan['scale']} (src->tgt)")
            axes[0, idx].axis('off')

            ax = axes[1, idx]
            ax.imshow(mod_mag, cmap='viridis')
            self._draw_rect(ax, plan['target_rect'], 'cyan', 'Target')
            ax.set_title(f"Target Feature Map: {plan['name']}")
            ax.set_xlabel('W')
            ax.set_ylabel('H')

        plt.tight_layout()

        compare_path = self.output_dir / f'{prefix}_feature_replace_inference_comparison.png'
        fig.savefig(compare_path, dpi=160, bbox_inches='tight')
        plt.close(fig)

        print('[6/6] 保存元信息...')
        metadata = {
            'source_sample_idx': self.source_sample_idx,
            'target_sample_idx': self.target_sample_idx,
            'fusion_method': self.fusion_method,
            'score_threshold': self.score_threshold,
            'max_transforms': self.max_transforms,
            'source_feature_shape': list(source_fused.shape),
            'target_feature_shape': list(target_baseline_fused.shape),
            'selected_source_pred_index': source_idx,
            'source_rect_yx': {
                'y0': int(source_rect[0]),
                'y1': int(source_rect[1]),
                'x0': int(source_rect[2]),
                'x1': int(source_rect[3]),
            },
            'source_baseline_vis': str(source_vis_path),
            'target_baseline_vis': str(target_base_vis_path),
            'modified_variants': [
                {
                    'name': item['plan']['name'],
                    'op': item['plan']['op'],
                    'scale': item['plan']['scale'],
                    'target_rect_yx': {
                        'y0': int(item['plan']['target_rect'][0]),
                        'y1': int(item['plan']['target_rect'][1]),
                        'x0': int(item['plan']['target_rect'][2]),
                        'x1': int(item['plan']['target_rect'][3]),
                    },
                    'target_hw': {
                        'h': int(item['plan']['target_hw'][0]),
                        'w': int(item['plan']['target_hw'][1]),
                    },
                    'modified_vis': str(variant_vis_paths[idx])
                }
                for idx, item in enumerate(variant_results)
            ],
            'comparison_vis': str(compare_path),
        }

        metadata_path = self.output_dir / 'feature_replace_test_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print('\n=== 跨样本替换测试完成 ===')
        print(f'source baseline可视化: {source_vis_path}')
        print(f'target baseline可视化: {target_base_vis_path}')
        for vis_path in variant_vis_paths:
            print(f'modified可视化: {vis_path}')
        print(f'对比图: {compare_path}')
        print(f'元信息: {metadata_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Replace fused feature region and compare inference results')
    parser.add_argument('--model', required=True, type=str, help='Path to checkpoint .pth')
    parser.add_argument('--config', required=True, type=str, help='Path to config yaml')
    parser.add_argument('--output', required=True, type=str, help='Output directory')
    parser.add_argument('--fusion_method', default='intermediate',
                        choices=['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'])
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--sample_idx', default=0, type=int,
                        help='Target sample index in test dataset')
    parser.add_argument('--source_sample_idx', default=None, type=int,
                        help='Source sample index for patch extraction. If not set, uses target sample index')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='Min score to choose source bbox, fallback to best bbox if none satisfied')
    parser.add_argument('--max_transforms', default=6, type=int,
                        help='Maximum number of transform variants for replacement test')
    return parser.parse_args()


def main():
    args = parse_args()

    tester = FusedFeatureRegionReplaceTester(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        fusion_method=args.fusion_method,
        device=args.device,
        sample_idx=args.sample_idx,
        source_sample_idx=args.source_sample_idx,
        score_threshold=args.score_threshold,
        max_transforms=args.max_transforms,
    )
    tester.run()


if __name__ == '__main__':
    main()
