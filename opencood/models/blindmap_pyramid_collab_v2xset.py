# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 17:02:54
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 17:02:54


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.pyramid_fuse_blindmap import BlindmapPyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
import opencood.tools.inference_runtime_config as runtime_config
class BlindmapPyramidCollabV2xset(nn.Module):
    def __init__(self, args):
        super(BlindmapPyramidCollabV2xset, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        # self.meter_per_pixel = args['m1']['encoder_args']['voxel_size'][0]
        self.cam_crop_info = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """

        Fusion, by default multiscale fusion:
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        # self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        self.pyramid_backbone = BlindmapPyramidFusion(args['fusion_backbone'])
        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)
    def forward(self, data_dict):
        if getattr(self, "force_collab", False):
            return self.forward_colla(data_dict)
        return self.forward_single(data_dict)
    def forward_colla(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        # ====== 记录输入到heter_feature_2d的时间 ======
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        # import time
        # t0 = time.time()
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True)

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue

            # start.record()
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            # end.record()
            # torch.cuda.synchronize()
            # print(f"encoder_{modality_name} time: {start.elapsed_time(end)}")
            # print(f"encoder_{modality_name} output shape: {feature.shape}")
            # encoder_m1 output shape: torch.Size([sum(CAV), 64, 256, 512])
            # start.record()
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            # end.record()
            # torch.cuda.synchronize()
            # print(f"backbone_{modality_name} time: {start.elapsed_time(end)}")
            # print(f"encoder_{modality_name} output shape: {feature.shape}")
            # encoder_m1 output shape: torch.Size([sum(CAV, 64, 128, 256])

            # start.record()
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

            # end.record()
            # torch.cuda.synchronize()
            # print(f"aligner_{modality_name} time: {start.elapsed_time(end)}")
            # print('modality_feature_dict.keys: ', modality_feature_dict.keys())
            # modality_feature_dict.keys:  dict_keys(['m1'])
        # ====== 记录输入到heter_feature_2d的时间 ======
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        # t1 = time.time()
        # print('time from encoder to heter_feature_2d: ', t1-t0)
        end.record()
        torch.cuda.synchronize()
        # print('time from encoder to heter_feature_2d: ', start.elapsed_time(end))
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)
        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # print(f"heter_feature_2d shape: {heter_feature_2d.shape}")
        # heter_feature_2d shape: torch.Size([4, 64, 128, 256])
        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        history_blind_maps = data_dict.get('history_blind_maps', None)
        # print(f"history_blind_maps shape : {history_blind_maps.shape if history_blind_maps is not None else 'None'}")
        fused_feature, communication_rates, batch_blind_maps, occ_outputs = self.pyramid_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len,
                                                affine_matrix,
                                                agent_modality_list,
                                                self.cam_crop_info,
                                                history_blind_maps=history_blind_maps
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        # # ============= 可视化部分 =============
        # visualize = True
        # vis_channels = [0, 4, 12, 21, 32, 45, 57, 60]
        # if visualize and not self.training:
        #     visualizer = CollaborativeFeatureVisualizer(
        #         save_dir=os.path.join(runtime_config.saved_path, './visualization_results')
        #     )

        # # 对每个batch样本进行可视化
        # for batch_idx in range(len(record_len)):
        #     visualizer.visualize_channel_features(
        #         heter_feature_2d=heter_feature_2d,
        #         fused_feature=fused_feature,
        #         agent_modality_list=agent_modality_list,
        #         record_len=record_len,
        #         channel_indices=vis_channels,
        #         sample_idx=batch_idx
        #     )

        #     # 计算并打印特征相似度
        #     start_idx = sum(record_len[:batch_idx])
        #     end_idx = start_idx + record_len[batch_idx]
        #     sample_heter = heter_feature_2d[start_idx:end_idx]
        #     sample_fused = fused_feature[batch_idx]

        #     similarity_metrics = visualizer.compute_feature_similarity(
        #         sample_heter, sample_fused
        #     )

        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        # detection_start_time = time.time()
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        # detection_end_time = time.time()
        # print(f"Detection time: {(detection_end_time - detection_start_time) * 1000:.2f} ms | batch size: {fused_feature.shape[0]} | batch fatures shape: {fused_feature.shape}")
        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        output_dict.update({'occ_single_list':
                            occ_outputs})
        output_dict.update({'comm_rate':
                            communication_rates})

        output_dict.update({'pred_blind_maps':
                            batch_blind_maps})

        return output_dict
    def forward_single(self, data_dict):
        output_dict = {'pyramid': 'single'}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True)

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue

            # start.record()
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)

            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']

            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        end.record()
        torch.cuda.synchronize()
        # print('time from encoder to heter_feature_2d: ', start.elapsed_time(end))
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)
        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features (ego-only for single agent forward)
        """
        # For ego-only forward, only assemble the first agent (ego)
        ego_modality = agent_modality_list[0]
        heter_feature_2d_list = [modality_feature_dict[ego_modality][0]]
        heter_feature_2d = torch.stack(heter_feature_2d_list)

        fused_feature, occ_outputs = self.pyramid_backbone.forward_single(
                                                heter_feature_2d
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)


        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        output_dict.update({'occ_single_list':
                            occ_outputs})

        return output_dict

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

class CollaborativeFeatureVisualizer:
    """
    可视化协同感知中的异构特征，用于验证特征对齐的必要性
    """
    def __init__(self, save_dir='./feature_visualization'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_channel_features(self, heter_feature_2d, fused_feature,
                                   agent_modality_list, record_len,
                                   channel_indices=None, sample_idx=0):
        """
        可视化ego和协同车辆的特征通道，以及融合后的特征

        Args:
            heter_feature_2d: [N, C, H, W] 融合前的异构特征
            fused_feature: [B, C, H, W] 融合后的特征
            agent_modality_list: 每个agent的模态类型列表
            record_len: 每个batch中agent数量的列表
            channel_indices: 要可视化的通道索引列表，如果为None则可视化前8个通道
            sample_idx: batch中要可视化的样本索引
        """
        if channel_indices is None:
            channel_indices = list(range(min(8, heter_feature_2d.shape[1])))

        # 计算当前sample的agent起始索引
        start_idx = sum(record_len[:sample_idx])
        end_idx = start_idx + record_len[sample_idx]

        # 提取当前sample的特征
        sample_heter_features = heter_feature_2d[start_idx:end_idx]  # [num_agents, C, H, W]
        sample_fused_feature = fused_feature[sample_idx]  # [C, H, W]
        sample_modalities = agent_modality_list[start_idx:end_idx]

        num_agents = len(sample_heter_features)
        num_channels = len(channel_indices)

        # 为每个通道创建可视化
        for ch_idx in channel_indices:
            self._visualize_single_channel(
                sample_heter_features, sample_fused_feature,
                sample_modalities, ch_idx, sample_idx
            )

        # 创建综合对比图
        self._visualize_channel_comparison(
            sample_heter_features, sample_fused_feature,
            sample_modalities, channel_indices, sample_idx
        )

    def _visualize_single_channel(self, heter_features, fused_feature,
                                  modalities, channel_idx, sample_idx):
        """可视化单个通道的所有agent特征和融合特征"""
        num_agents = len(heter_features)

        fig = plt.figure(figsize=(20, 4))
        gs = GridSpec(1, num_agents + 2, figure=fig, wspace=0.3)

        # 可视化每个agent的特征
        for i in range(num_agents):
            ax = fig.add_subplot(gs[0, i])
            feature_map = heter_features[i, channel_idx].cpu().detach().numpy()

            im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
            agent_type = 'Ego' if i == 0 else f'Agent{i}'
            ax.set_title(f'{agent_type}\n({modalities[i]})', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 可视化融合后的特征
        ax_fused = fig.add_subplot(gs[0, num_agents])
        fused_map = fused_feature[channel_idx].cpu().detach().numpy()
        im = ax_fused.imshow(fused_map, cmap='viridis', aspect='auto')
        ax_fused.set_title('Fused Feature', fontsize=10, fontweight='bold')
        ax_fused.axis('off')
        plt.colorbar(im, ax=ax_fused, fraction=0.046, pad=0.04)

        # 添加差异热图（ego vs fused）
        ax_diff = fig.add_subplot(gs[0, num_agents + 1])
        ego_map = heter_features[0, channel_idx].cpu().detach().numpy()
        diff_map = np.abs(ego_map - fused_map)
        im = ax_diff.imshow(diff_map, cmap='hot', aspect='auto')
        ax_diff.set_title('|Ego - Fused|', fontsize=10, fontweight='bold')
        ax_diff.axis('off')
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

        plt.suptitle(f'Channel {channel_idx} Feature Comparison (Sample {sample_idx})',
                     fontsize=14, fontweight='bold', y=1.02)

        save_path = os.path.join(self.save_dir, f'sample{sample_idx}_channel{channel_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_path}')

    def _visualize_channel_comparison(self, heter_features, fused_feature,
                                     modalities, channel_indices, sample_idx):
        """创建多通道综合对比图"""
        num_agents = len(heter_features)
        num_channels = len(channel_indices)

        fig = plt.figure(figsize=(20, 4 * num_channels))
        gs = GridSpec(num_channels, num_agents + 2, figure=fig,
                     hspace=0.3, wspace=0.3)

        for row, ch_idx in enumerate(channel_indices):
            # 可视化每个agent
            for col in range(num_agents):
                ax = fig.add_subplot(gs[row, col])
                feature_map = heter_features[col, ch_idx].cpu().detach().numpy()

                im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
                if row == 0:
                    agent_type = 'Ego' if col == 0 else f'Agent{col}'
                    ax.set_title(f'{agent_type} ({modalities[col]})', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'Ch {ch_idx}', fontsize=10, fontweight='bold')
                ax.axis('off')

                if col == 0:  # 只为第一列添加colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 融合特征
            ax_fused = fig.add_subplot(gs[row, num_agents])
            fused_map = fused_feature[ch_idx].cpu().detach().numpy()
            im = ax_fused.imshow(fused_map, cmap='viridis', aspect='auto')
            if row == 0:
                ax_fused.set_title('Fused', fontsize=10, fontweight='bold')
            ax_fused.axis('off')
            plt.colorbar(im, ax=ax_fused, fraction=0.046, pad=0.04)

            # 差异图
            ax_diff = fig.add_subplot(gs[row, num_agents + 1])
            ego_map = heter_features[0, ch_idx].cpu().detach().numpy()
            diff_map = np.abs(ego_map - fused_map)
            im = ax_diff.imshow(diff_map, cmap='hot', aspect='auto')
            if row == 0:
                ax_diff.set_title('|Ego-Fused|', fontsize=10, fontweight='bold')
            ax_diff.axis('off')
            plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

        plt.suptitle(f'Multi-Channel Feature Comparison (Sample {sample_idx})',
                     fontsize=16, fontweight='bold', y=0.995)

        save_path = os.path.join(self.save_dir,
                                f'sample{sample_idx}_multi_channel_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_path}')

    def compute_feature_similarity(self, heter_features, fused_feature):
        """
        计算特征相似度指标

        Returns:
            dict: 包含各种相似度指标的字典
        """
        num_agents = len(heter_features)
        ego_feature = heter_features[0]  # [C, H, W]

        # 计算ego与其他agent的相似度
        cosine_sims = []
        mse_values = []

        for i in range(1, num_agents):
            # Cosine similarity
            ego_flat = ego_feature.flatten()
            agent_flat = heter_features[i].flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                ego_flat.unsqueeze(0), agent_flat.unsqueeze(0)
            ).item()
            cosine_sims.append(cos_sim)

            # MSE
            mse = torch.nn.functional.mse_loss(ego_feature, heter_features[i]).item()
            mse_values.append(mse)

        # 计算ego与融合特征的相似度
        ego_fused_cos = torch.nn.functional.cosine_similarity(
            ego_feature.flatten().unsqueeze(0),
            fused_feature.flatten().unsqueeze(0)
        ).item()

        ego_fused_mse = torch.nn.functional.mse_loss(ego_feature, fused_feature).item()

        return {
            'inter_agent_cosine_sim': cosine_sims,
            'inter_agent_mse': mse_values,
            'ego_fused_cosine_sim': ego_fused_cos,
            'ego_fused_mse': ego_fused_mse,
            'avg_inter_agent_cosine': np.mean(cosine_sims) if cosine_sims else 0,
            'avg_inter_agent_mse': np.mean(mse_values) if mse_values else 0
        }
