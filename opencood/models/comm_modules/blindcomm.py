# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:27:06
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:27:06

import torch
import torch.nn as nn
import numpy as np


def calc_comm_mask(comm_maps, comm_volume_MB, H, W, bev_channels_list):
    """
    only use when level == 1
    Args:
        comm_maps: [N, 1, H, W]
        comm_volume_MB: comm volume limit (MB)
        H: height of BEV feature
        W: width of BEV feature
        bev_channels: BEV feature channels
      # feature shape torch.Size([4, 64, 128, 256])
        # feature shape torch.Size([4, 128, 64, 128])
        # feature shape torch.Size([4, 256, 32, 64])
    Return:
        comm_mask: [N, 1, H, W]
    """
    if len(bev_channels_list) != 3:
        full_B = H * W * bev_channels_list[0] * 4
    else:
        full_B = H * W * bev_channels_list[0] * 4 + 0.5* H * 0.5* W * bev_channels_list[1] * 4 + 0.25* H * 0.25* W * bev_channels_list[2] * 4
    # full_B = H * W * bev_channels_list[0] * 4
    comm_rate = min(1.0, comm_volume_MB * 1024 * 1024 / full_B)
    k = max(1, int(comm_rate * H * W))
    comm_mask = torch.zeros_like(comm_maps)
    for i in range(comm_maps.shape[0]):
        if i != 0:
            flat_map = comm_maps[i].view(-1)
            _, indices = torch.topk(flat_map, k=min(k, flat_map.numel()))
            flat_mask = torch.zeros_like(flat_map)
            flat_mask[indices] = 1
            comm_mask[i] = flat_mask.view(comm_maps[i].shape)

    return comm_mask, comm_rate


def calc_comm_mask_total_budget(comm_maps, comm_volume_MB, H, W, bev_channels_list):
    # Scene-level budget: comm_volume_MB is shared by all non-ego agents.
    if len(bev_channels_list) != 3:
        per_agent_full_B = H * W * bev_channels_list[0] * 4
    else:
        per_agent_full_B = (
            H * W * bev_channels_list[0] * 4
            + 0.5 * H * 0.5 * W * bev_channels_list[1] * 4
            + 0.25 * H * 0.25 * W * bev_channels_list[2] * 4
        )

    num_collab_agents = max(comm_maps.shape[0] - 1, 0)
    if num_collab_agents == 0:
        return torch.zeros_like(comm_maps), 0.0

    total_full_B = per_agent_full_B * num_collab_agents
    comm_rate = min(1.0, comm_volume_MB * 1024 * 1024 / total_full_B)
    k = max(1, int(comm_rate * H * W))

    comm_mask = torch.zeros_like(comm_maps)
    for i in range(1, comm_maps.shape[0]):
        flat_map = comm_maps[i].view(-1)
        _, indices = torch.topk(flat_map, k=min(k, flat_map.numel()))
        flat_mask = torch.zeros_like(flat_map)
        flat_mask[indices] = 1
        comm_mask[i] = flat_mask.view(comm_maps[i].shape)

    return comm_mask, comm_rate


class BlindCommunication(nn.Module):
    def __init__(self, args, channels_list):
        super(BlindCommunication, self).__init__()

        self.smooth = False
        self.thre = args["thre"]
        if "gaussian_smooth" in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

        self.use_threshold = (
            True if "use_threshold" not in args else args["use_threshold"]
        )
        print('use_threshold:', self.use_threshold)
        self.comm_volume_MB = (
            1
            if "comm_volume_MB" not in args and not self.use_threshold
            else args["comm_volume_MB"]
        )
        # self.bev_channels = (
        #     256
        #     if "bev_channels" not in args and not self.use_threshold
        #     else args["bev_channels"]
        # )
        self.bev_channels_list = channels_list

        if not self.use_threshold:
            print(
                f"Communication Module: comm_volume_MB={self.comm_volume_MB}"
            )
        else:
            print(f"Communication Module: thre={self.thre}")

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = (
                1
                / (2 * np.pi * sigma)
                * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            )
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = (
            torch.Tensor(gaussian_kernel)
            .to(self.gaussian_filter.weight.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_blind_maps_groups, record_len, pairwise_t_matrix):
        # batch_confidence_maps:([L1, H, W], [L2, H, W], ...)
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        # print("comm_volume_MB when inferencing :{}".format(self.comm_volume_MB))
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_blind_maps_groups[0].shape

        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        # print(f'*****************{__file__}*****************')
        # print('batch_blind_maps_groups[0].shape:', batch_blind_maps_groups[0].shape)
        # torch.Size([2, 1, 128, 256]
        
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # 处理所有agent的maps
            all_communication_maps = []
            # print('batch_blind_maps_groups[b][0] shape:', batch_blind_maps_groups[b][0].shape)
            # print('batch_blind_maps_groups[b][0].sigmoid() shape:', batch_blind_maps_groups[b][0].sigmoid().shape)
            # print('batch_blind_maps_groups[b][0].sigmoid().unsqueeze(1) shape:', batch_blind_maps_groups[b][0].sigmoid().unsqueeze(1).shape)
            # batch_blind_maps_groups[b][0] shape: torch.Size([1, 100, 252])
            # batch_blind_maps_groups[b][0].sigmoid() shape: torch.Size([1, 100, 252])
            # batch_blind_maps_groups[b][0].sigmoid().unsqueeze(1) shape: torch.Size([1, 1, 100, 252])
            for i in range(N):
                # if i == N-1:  # 只对最后一个agent进行sigmoid
                #     agent_map = batch_blind_maps_groups[b][i].sigmoid().unsqueeze(1)
                # else:
                #     agent_map = batch_blind_maps_groups[b][i].unsqueeze(1)
                # all_communication_maps.append(agent_map)
                # ——————————————batch_blind_maps_groups通过最后sigmoid的网络得到，不需要再sigmoid——————————————————
                agent_map = batch_blind_maps_groups[b][i].unsqueeze(1)
                # print('agent_map shape:', agent_map.shape)
                all_communication_maps.append(agent_map)

            # 拼接所有agent的maps
            infra_communication_maps = torch.cat(all_communication_maps, dim=0)
            # print('infra_communication_maps shape:', infra_communication_maps.shape)
            # 推理时：torch.Size([1, 1, 128, 256])
            if self.smooth:
                communication_maps = self.gaussian_filter(infra_communication_maps)
            else:
                communication_maps = infra_communication_maps
            # communication_maps(L, 1, H, W)
            if self.use_threshold:
                ones_mask = torch.ones_like(communication_maps).to(
                    communication_maps.device
                )
                zeros_mask = torch.zeros_like(communication_maps).to(
                    communication_maps.device
                )
                communication_mask = torch.where(
                    communication_maps > self.thre, ones_mask, zeros_mask
                )
                # print('communication_mask shape:', communication_mask.shape)
                # communication_mask shape: torch.Size([2, 1, 128, 256])

                ######################可视化mask以检查######################
                # communication_maps_before_smooth = infra_communication_maps[1].clone()
                # communication_maps = communication_maps[1].clone()
                # communication_mask_infra = communication_mask[1].clone()
                #  # shape (100, 252)
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(15,5))
                # plt.subplot(1,3,1)
                # plt.imshow(communication_maps_before_smooth[0].cpu().numpy(), cmap='gray')
                # plt.title('infra comm map')
                # plt.subplot(1,3,2)
                # plt.imshow(communication_maps[0].cpu().numpy(), cmap='gray')
                # plt.title('infra comm map after smooth')
                # plt.subplot(1,3,3)
                # plt.imshow(communication_mask_infra[0].cpu().numpy(), cmap='gray')
                # plt.title('Infra Mask')
                # plt.show()
                ######################可视化mask以检查######################

                communication_rate = communication_mask[-1].sum() / (H * W)
            else:
                communication_mask, communication_rate = calc_comm_mask(
                    communication_maps, self.comm_volume_MB, H, W, self.bev_channels_list
                )
            

            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(
                communication_mask.device
            )
            # Only the first agent is ego; all collaborators follow the communication mask.
            communication_mask_nodiag[0] = ones_mask[0]

            ######################可视化mask以检查######################
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10,5))
            # plt.subplot(1,2,1)
            # plt.imshow(communication_mask_nodiag[0][0].cpu().numpy(), cmap='gray', vmin=0,   vmax=1)
            # plt.colorbar()
            # plt.title('Ego Mask')
            # plt.subplot(1,2,2)
            # plt.imshow(communication_mask_nodiag[1][0].cpu().numpy(), cmap='gray', vmin=0,   vmax=1)
            # plt.colorbar()
            # plt.title('Infra Mask')
            # plt.show()
            ######################可视化mask以检查######################

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(
                communication_maps * communication_mask_nodiag
            )
        communication_rates = sum(communication_rates) / B
        # print('communication_rates after batch:', communication_rates)
        communication_masks = torch.concat(communication_masks, dim=0)
        return (
            batch_communication_maps,
            communication_masks,
            communication_rates,
        )
