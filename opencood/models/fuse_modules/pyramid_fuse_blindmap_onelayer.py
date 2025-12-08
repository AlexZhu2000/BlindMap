# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature
from opencood.models.comm_modules.blindmap import BlindMap
from opencood.models.comm_modules.blindcomm import BlindCommunication
import torch.nn.functional as F
import os
from datetime import datetime
import matplotlib.pyplot as plt
def visualize_blind_maps(
    batch_blind_maps, record_len, save_path=None, show=True, show_axis=False
):
    """
    Directly visualize blind maps without modifying values
    Args:
        batch_blind_maps: (sum(n_cav), 1, H, W) tensor
        record_len: list of number of agents per batch
        save_path: path to save visualization
        show: whether to display plot
        show_axis: whether to show axis in plot
    """
    B = len(record_len)
    batch_blind_maps = batch_blind_maps.detach().cpu().numpy()

    # 创建保存目录
    if save_path is None:
        save_dir = "./vis/blindmap"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"blind_maps_{timestamp}.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建图像
    fig, axes = plt.subplots(B, 2, figsize=(12, 3 * B), dpi=300)
    fig.suptitle("Blind Maps Visualization", fontsize=16)

    # 处理B=1的特殊情况
    if B == 1:
        axes = np.array([axes])

    for b in range(B):
        if record_len[b] < 2:
            continue
        # Ego map
        ego_map = batch_blind_maps[b * 2, 0]
        # im1 = axes[b, 0].imshow(ego_map, cmap="viridis", vmin=0, vmax=1)
        im1 = axes[b, 0].imshow(ego_map)
        axes[b, 0].set_title(f"Batch {b + 1} - Ego")
        if not show_axis:
            axes[b, 0].axis("off")  # 移除坐标轴以获得更清晰的可视化
        cbar1 = plt.colorbar(im1, ax=axes[b, 0])
        cbar1.set_label("Blind Probability")

        # Infra map
        infra_map = batch_blind_maps[b * 2 + 1, 0]
        # im2 = axes[b, 1].imshow(infra_map, cmap="viridis", vmin=0, vmax=1)
        im2 = axes[b, 1].imshow(infra_map)
        axes[b, 1].set_title(f"Batch {b + 1} - Infra")
        if not show_axis:
            axes[b, 1].axis("off")
        cbar2 = plt.colorbar(im2, ax=axes[b, 1])
        cbar2.set_label("Blind Probability")

    plt.tight_layout()

    # 保存为SVG格式
    plt.savefig(save_path, format="png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out
class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]
    

class SumFusion(nn.Module):
    def __init__(self):
        super(SumFusion, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=0)
    

class BlindmapPyramidFusionAlation(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        print('BlindmapPyramidFusion')
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.blindmap = BlindMap(model_cfg['blindmap'], self.model_cfg['num_filters'][0])
        self.naive_communication = BlindCommunication(self.model_cfg["communication"], self.model_cfg['num_filters'])
        self.downsample_rate = 2
        self.bev_H = 128
        self.bev_W = 256
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        self.agg_mode = self.model_cfg["communication"].get('fusion_mode', "MAX")
        
        self.fuse_modules = nn.ModuleList()
        for idx in range(self.num_levels):
            if self.agg_mode == "MAX":
                fuse_network = MaxFusion()
            elif self.agg_mode == "SUM":
                fuse_network = SumFusion()
            self.fuse_modules.append(fuse_network)
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list

    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        # print('record_len', record_len)
        B, L = affine_matrix.shape[:2]
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)
        # for feature in feature_list:
        #     print("feature shape",feature.shape)
        # feature shape torch.Size([4, 64, 128, 256])
        # feature shape torch.Size([4, 128, 64, 128])
        # feature shape torch.Size([4, 256, 32, 64])
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            '''
            使用最高层、最大分辨率的特征图进行blindmap预测，其它层使用resize版本
            '''
            x = feature_list[i]
            _, c, curr_H, curr_W = x.shape
            # print('x.shape:', x.shape)
            # torch.Size([4, 64, 128, 256]
            if i==0:
                rate = self.bev_H / curr_H
                pixel_size = 0.4 * rate
                import time
                start_time = time.time()
                batch_blind_maps = self.blindmap(
                                    x, record_len, affine_matrix, pixel_size
                                )
                end_time = time.time()
                single_batch_time = (end_time - start_time) / B 
                # print(f"Blindmap time: {single_batch_time * 1000:.2f} ms | batch size: {B} | batch fatures shape: {x.shape}")
                # 1.2ms
                # print('batch_blind_maps.shape:', batch_blind_maps.shape)
                # torch.Size([4, 1, 128, 256]
                # visualize_blind_maps(batch_blind_maps, record_len)
                batch_blind_maps_groups = regroup(
                            batch_blind_maps, record_len
                        )
                (batch_communication_maps_list,
                            communication_masks,
                            communication_rates,
                        ) = self.naive_communication(
                            batch_blind_maps_groups, record_len, affine_matrix
                        )
                original_maps = torch.cat(batch_communication_maps_list, dim=0)
                x = x * communication_masks
            else:
                continue
            # elif i > 0:
            #         # Other layers: resize communication maps
            #         resized_maps = F.interpolate(
            #             original_maps,
            #             size=(curr_H, curr_W),
            #             mode='bilinear',
            #             align_corners=True
            #                 )
            #         resized_mask = F.interpolate(
            #             communication_masks,
            #             size=(curr_H, curr_W),
            #             mode='bilinear',
            #             align_corners=True
            #                 )
            #         x = x * resized_mask
            batch_node_features = regroup(x, record_len)
            batch_communication_maps = regroup( original_maps, 
                        record_len
                    )
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = affine_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                # print('node_features.shape:', node_features.shape)
                # torch.Size([2, 64, 128, 256]
                # node_features torch.Size([2, c, 100, 252]
                C, H, W = node_features.shape[1:]
                # print("t_matrix ego ---> infra:",t_matrix[0, 1, :, :])
                # [0, 0]是ego的坐标,无转换
                neighbor_feature = warp_affine_simple(
                    node_features, t_matrix[0, :, :, :], (H, W)
                )
                # 根据融合模式选择不同的融合方法
                if self.agg_mode == "Blindfusion":
                    # 使用当前层对应分辨率的通信图
                    curr_comm_maps = batch_communication_maps[b]
                    # print('curr_comm_maps.shape:', curr_comm_maps.shape)
                    # torch.Size([2, 1, 100, 252]
                    neighbor_comm_maps = warp_affine_simple(
                        curr_comm_maps,
                        t_matrix[0, :, :, :],
                        (H, W),  # 使用当前层的分辨率
                        mode='bilinear'
                    )
                    x_fuse.append(self.fuse_modules[i](neighbor_feature, neighbor_comm_maps))
                else:
                    # 其他层使用max融合
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
            x_fuse = torch.stack(x_fuse)
            # print('x_fuse.shape:', x_fuse.shape)
            fused_feature_list.append(x_fuse)

            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            # score = torch.sigmoid(occ_map) + 1e-4

            # if crop_mask_flag and not self.training:
            #     cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
            #     _, _, H, W = cam_crop_mask.shape
            #     for cam_modality in cam_modality_set:
            #         crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
            #         crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

            #         start_h = int(H//2-crop_H//2)
            #         end_h = int(H//2+crop_H//2)
            #         start_w = int(W//2-crop_W//2)
            #         end_w = int(W//2+crop_W//2)

            #         cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
            #         cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

            #     score = score * cam_crop_mask

            # fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        return fused_feature, communication_rates, batch_blind_maps, occ_map_list