# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np


def calc_comm_mask_total_budget(comm_maps, comm_volume_MB, bev_channels_list):
    """
    Select Where2comm confidence regions under a scene-level budget.

    The budget is shared by all non-ego collaborators. Region selection is
    performed on the highest-resolution map and the caller reuses/resizes the
    mask for lower pyramid levels.
    """
    _, _, H, W = comm_maps.shape

    if not bev_channels_list:
        per_agent_full_bytes = H * W * 4
    elif len(bev_channels_list) != 3:
        per_agent_full_bytes = H * W * bev_channels_list[0] * 4
    else:
        per_agent_full_bytes = (
            H * W * bev_channels_list[0] * 4
            + 0.5 * H * 0.5 * W * bev_channels_list[1] * 4
            + 0.25 * H * 0.25 * W * bev_channels_list[2] * 4
        )

    num_collab_agents = max(comm_maps.shape[0] - 1, 0)
    if num_collab_agents == 0:
        return torch.zeros_like(comm_maps), 0.0

    total_full_bytes = per_agent_full_bytes * num_collab_agents
    comm_rate = min(1.0, comm_volume_MB * 1024 * 1024 / total_full_bytes)
    k = max(1, int(comm_rate * H * W))

    comm_mask = torch.zeros_like(comm_maps)
    for i in range(1, comm_maps.shape[0]):
        flat_map = comm_maps[i].reshape(-1)
        _, indices = torch.topk(flat_map, k=min(k, flat_map.numel()))
        flat_mask = torch.zeros_like(flat_map)
        flat_mask[indices] = 1
        comm_mask[i] = flat_mask.view_as(comm_maps[i])

    return comm_mask, comm_rate


class Communication(nn.Module):
    def __init__(self, args, channels_list=None):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        self.use_threshold = args.get('use_threshold', True)
        self.comm_volume_MB = args.get('comm_volume_MB', 1)
        self.bev_channels_list = channels_list
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            if self.use_threshold:
                communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)

                if N > 1:
                    communication_rate = communication_mask[1:N].sum()/((N - 1) * H * W)
                else:
                    communication_rate = torch.zeros(1, device=communication_maps.device)[0]
            else:
                communication_mask, communication_rate = calc_comm_mask_total_budget(
                    communication_maps, self.comm_volume_MB, self.bev_channels_list
                )

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            # Only the first CAV is ego in the OpenCOOD intermediate-fusion batch.
            communication_mask_nodiag[0] = ones_mask[0]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
