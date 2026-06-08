# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

def calc_comm_mask_multilevel(scores_list, comm_volume_MB, bev_channels_list):
    """
    Calculate communication masks using the same accounting as BlindMap.

    A single high-resolution BEV mask is selected from level-0 confidence
    scores, while the communication budget accounts for all pyramid feature
    levels. The selected mask is resized to lower-resolution levels.
    """
    base_score = scores_list[0]
    N, _, H, W = base_score.shape

    if len(bev_channels_list) != len(scores_list):
        per_agent_full_bytes = H * W * bev_channels_list[0] * 4
    else:
        per_agent_full_bytes = 0
        for level_idx, score in enumerate(scores_list):
            _, _, level_h, level_w = score.shape
            per_agent_full_bytes += level_h * level_w * bev_channels_list[level_idx] * 4

    num_collab_agents = max(N - 1, 0)
    if num_collab_agents == 0:
        comm_rate = 0.0
    else:
        total_full_bytes = per_agent_full_bytes * num_collab_agents
        comm_rate = min(1.0, comm_volume_MB * 1024 * 1024 / total_full_bytes)
    k = max(1, int(comm_rate * H * W))

    base_mask = torch.zeros_like(base_score)
    base_mask[0] = torch.ones_like(base_mask[0])
    for agent_idx in range(1, N):
        flat_score = base_score[agent_idx].reshape(-1)
        _, indices = torch.topk(flat_score, k=min(k, flat_score.numel()))
        flat_mask = torch.zeros_like(flat_score)
        flat_mask[indices] = 1
        base_mask[agent_idx] = flat_mask.view_as(base_score[agent_idx])

    masks_list = [base_mask]
    for score in scores_list[1:]:
        resized_mask = F.interpolate(
            base_mask,
            size=score.shape[-2:],
            mode="nearest",
        )
        masks_list.append(resized_mask)

    return masks_list, comm_rate
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

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
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
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        self.comm_volume_MB = model_cfg.get('comm_volume_MB', None)
        if self.comm_volume_MB is not None:
            print(f'Communication volume limit: {self.comm_volume_MB} MB')
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
        # Get BEV channels list for communication volume calculation
        bev_channels_list = self.model_cfg["num_filters"] if self.comm_volume_MB is not None else None
        fused_feature_list = []
        occ_map_list = []
        score_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask
            score_list.append(score)
        # Calculate communication masks based on total volume limit across all levels
        comm_masks_list = None
        comm_rate = 1.0
        if self.comm_volume_MB is not None and bev_channels_list is not None:
            split_score_list = [regroup(score, record_len) for score in score_list]
            
            # For each batch
            all_comm_masks_list = []
            all_comm_rates = []
            
            for b in range(len(record_len)):
                batch_scores = [split_scores[b] for split_scores in split_score_list]
                batch_masks, batch_rate = calc_comm_mask_multilevel(batch_scores, self.comm_volume_MB, bev_channels_list)
                all_comm_masks_list.append(batch_masks)
                all_comm_rates.append(batch_rate)
            
            comm_rate = np.mean(all_comm_rates)
            # print(f"Average communication rate: {comm_rate:.4f}")
            
        # Now fuse features for each level using the communication masks
        fused_feature_list = []
        for i in range(self.num_levels):
            if self.comm_volume_MB is not None and all_comm_masks_list is not None:
                # Apply communication mask to the scores
                masked_scores = []
                for b in range(len(record_len)):
                    start_idx = sum(record_len[:b])
                    end_idx = start_idx + record_len[b]
                    batch_masked_score = score_list[i][start_idx:end_idx] * all_comm_masks_list[b][i]
                    masked_scores.append(batch_masked_score)
                masked_score = torch.cat(masked_scores, dim=0)
            else:
                masked_score = score_list[i]
                
            # Fuse features
            fused_feature_list.append(weighted_fuse(
                feature_list[i], 
                masked_score, 
                record_len, 
                affine_matrix, 
                self.align_corners
            ))
            # fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        
        self.last_comm_rate = comm_rate
        return fused_feature, occ_map_list 