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

def calc_comm_mask_multilevel(scores_list, comm_volume_MB, bev_channels_list):
    """
    Calculate communication masks for all levels based on a total communication volume limit
    
    Args:
        scores_list: list of tensors, each of shape [N, 1, H_i, W_i] for level i
        comm_volume_MB: comm volume limit (MB)
        bev_channels_list: BEV feature channels [c1, c2, c3]
        # feature shape torch.Size([4, 64, 128, 256])
        # feature shape torch.Size([4, 128, 64, 128])
        # feature shape torch.Size([4, 256, 32, 64])
    Return:
        comm_masks_list: list of tensors, each of shape [N, 1, H_i, W_i] for level i
        comm_rate: overall communication rate
    """
    N = scores_list[0].shape[0]
    device = scores_list[0].device
    
    # Calculate sizes of features at each level
    sizes = []
    for i, score in enumerate(scores_list):
        _, _, H, W = score.shape
        # Calculate feature size in bytes (float32 = 4 bytes per element)
        level_size = H * W * bev_channels_list[i] * 4
        sizes.append(level_size)
    
    # Total size of all features
    total_size = sum(sizes)
    
    # Calculate overall communication rate based on volume limit
    comm_rate = min(1.0, comm_volume_MB * 1024 * 1024 / total_size)
    
    # Create masks for all levels
    masks_list = []
    
    # Calculate importance scores across all levels
    all_scores = []
    level_indices = []
    
    for level_idx, score in enumerate(scores_list):
        # 初始化该级别的掩码
        masks_list.append(torch.zeros_like(score))
        for i in range(N):
            if i % 2 != 0:  # Only for non-ego vehicles
                flat_score = score[i].view(-1)
                all_scores.append(flat_score)
                # Remember which level and which spatial position each score came from
                level_indices.append((level_idx, i, score[i].shape))
    
    # Concatenate all scores
    if all_scores:
        all_scores_tensor = torch.cat(all_scores)
        
        # Calculate how many elements to keep total
        total_elements = all_scores_tensor.numel()
        k_total = max(1, int(comm_rate * total_elements))
        
        # Get top k indices globally
        _, top_indices = torch.topk(all_scores_tensor, k=min(k_total, total_elements))
        
        # Create a mask for the flattened tensor
        global_mask = torch.zeros_like(all_scores_tensor)
        global_mask[top_indices] = 1
        
        # Split the mask back to individual levels and reshape
        start_idx = 0
        
        # Fill in the masks
        for (level_idx, agent_idx, original_shape) in level_indices:
            end_idx = start_idx + original_shape.numel()
            agent_mask = global_mask[start_idx:end_idx].view(original_shape)
            masks_list[level_idx][agent_idx] = agent_mask
            start_idx = end_idx
    else:
        # If no non-ego vehicles, create empty masks
        masks_list = [torch.zeros_like(score) for score in scores_list]
    
    # For ego vehicles (i==0), use full feature
    for level_idx, mask in enumerate(masks_list):
        for i in range(N):
            if i % 2 == 0:  # Ego vehicles
                masks_list[level_idx][i] = torch.ones_like(masks_list[level_idx][i])
    
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

        
        return fused_feature, occ_map_list 