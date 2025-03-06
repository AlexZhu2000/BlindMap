import torch
import torch.nn as nn
import numpy as np

C = 64
class BlindMap(nn.Module):
    def __init__(self, args):
        super(BlindMap, self).__init__()
        # 特征提取和融合网络
        self.conv1 = nn.Conv2d(C + 1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    def create_location_feature(self, ego_pillar_loc_in_agent, H, W):
        # 创建位置特征图
        B = len(ego_pillar_loc_in_agent)
        loc_features = torch.zeros((B, 1, H, W), device=ego_pillar_loc_in_agent[0].device)
        
        for i, loc in enumerate(ego_pillar_loc_in_agent):
            # 将归一化坐标转换为图像坐标
            x = ((loc[0] + 1) * W / 2).long().clamp(0, W-1)
            y = ((loc[1] + 1) * H / 2).long().clamp(0, H-1)
            # print('ego location in bev map:', x, y)
            loc_features[i, 0, y, x] = 1.0
            
        return loc_features
    def get_pillar_loc(self, B, record_len, pairwise_t_matrix):
        '''
        计算每个ego车辆在ego、infra坐标下的位置
        '''
        ego_pillar_loc_in_agent = []
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            loc_in_agents = []
            for i in range(N):
                M = t_matrix[0, i]
                        # 使用变换矩阵计算位置
                # [tx, ty] = M @ [0, 0, 1]
                tx = M[0, 2]  # 变换后的x坐标(归一化的)
                ty = M[1, 2]  # 变换后的y坐标(归一化的)
                # 将当前batch的坐标转换为tensor
                ego_pillar_loc_in_agent.append(torch.tensor([tx, ty], dtype=t_matrix.dtype, device=t_matrix.device))
                # 将归一化坐标转换为物理坐标
                # tx_physical = tx * (self.downsample_rate * self.discrete_ratio * W) / 2
                # ty_physical = ty * (self.downsample_rate * self.discrete_ratio * H) / 2
                # loc_in_agents.append([tx_physical, ty_physical])
        return ego_pillar_loc_in_agent
    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L, _, _, _ = pairwise_t_matrix.shape
        ego_pillar_loc_in_agent = self.get_pillar_loc(B, record_len, pairwise_t_matrix)
        ego_loc_features = self.create_location_feature(ego_pillar_loc_in_agent, H, W)
        # 3. 特征融合
        fused_features = torch.cat([x, ego_loc_features], dim=1)  # (sum(n_cav), C+1, H, W)

         # 4. 通过网络预测盲区
        out = self.relu(self.conv1(fused_features))
        out = self.relu(self.conv2(out))
        blind_map = self.sigmoid(self.conv3(out))  # (sum(n_cav), 1, H, W)
        
        return blind_map

        
        
