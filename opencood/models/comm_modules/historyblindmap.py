# @Author: Zhenhan Zhu (zhuzhenhan@nuaa.edu.cn)
# @Date: 2025-12-08 19:27:25
# @Last Modified by: Zhenhan Zhu
# @Last Modified time: 2025-12-08 19:27:25

import torch
import torch.nn as nn
import numpy as np

C = 64


class HistoryBlindMap(nn.Module):
    def __init__(self, model_cfg, in_channels=C, bev_H=128, bev_W=256):
        super(HistoryBlindMap, self).__init__()
        # RIPE related parameters
        self.use_ripe = model_cfg.get('use_ripe', True)
        self.ripe_dim = model_cfg.get('ripe_dim', 8)
        self.decay_lambda = model_cfg.get('decay_lambda', 10.0)


        # History related parameters
        self.hidden_dim = model_cfg.get('hidden_dim', 64)
        self.use_history = model_cfg.get('use_history', False)
        self.history_dim = model_cfg.get('history_dim', 64)
        self.history_num = model_cfg.get('history_num', 3)
        
        # print('use history blind map:', self.use_history)
        base_channels = in_channels + (self.ripe_dim if self.use_ripe else 1)
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # 历史特征处理网络
        if self.use_history:
            self.history_net = nn.Sequential(
                nn.Conv2d(1, self.history_dim, 3, padding=1),
                nn.BatchNorm2d(self.history_dim),
                nn.ReLU()
            )
            
            # 方案1: 使用3D卷积处理时序特征
            self.temporal_fusion_3d = nn.Sequential(
                nn.Conv3d(self.history_dim, self.history_dim, 
                         kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(self.history_dim),
                nn.ReLU(),
                nn.Conv3d(self.history_dim, self.hidden_dim, 
                         kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(self.hidden_dim),
                nn.ReLU()
            )
            
            # 方案2: 使用ConvLSTM (备选)
            # self.temporal_fusion_lstm = ConvLSTM(
            #     input_dim=self.history_dim,
            #     hidden_dim=self.hidden_dim,
            #     kernel_size=(3, 3),
            #     num_layers=1
            # )
            
            # 方案3: 使用注意力机制融合时序特征
            # self.temporal_attention = nn.Sequential(
            #     nn.Conv2d(self.history_dim, self.history_dim // 4, 1),
            #     nn.ReLU(),
            #     nn.Conv2d(self.history_dim // 4, 1, 1),
            #     nn.Sigmoid()
            # )
            # Update base_channels to include history features
            base_channels += self.hidden_dim
        # 输出层
        # Feature fusion and output networks
        self.conv1 = nn.Conv2d(base_channels, self.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def process_history_features_3d(self, history_blind_maps):
        """使用3D卷积处理历史特征"""
        B, T, H, W = history_blind_maps.shape
        
        # Process each history map
        history_feat = self.history_net(history_blind_maps.view(-1, 1, H, W))
        history_feat = history_feat.view(B, T, self.history_dim, H, W)
        
        # 3D convolution for temporal fusion
        # Input: (B, T, C, H, W) -> (B, C, T, H, W)
        history_feat = history_feat.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolution
        fused_feat = self.temporal_fusion_3d(history_feat)  # (B, hidden_dim, T, H, W)
        
        # Global average pooling over time dimension
        fused_feat = torch.mean(fused_feat, dim=2)  # (B, hidden_dim, H, W)
        
        return fused_feat
    def process_history_features_attention(self, history_blind_maps):
        """使用注意力机制处理历史特征"""
        B, T, H, W = history_blind_maps.shape
        
        # Process each history map
        history_feat = self.history_net(history_blind_maps.view(-1, 1, H, W))
        history_feat = history_feat.view(B, T, self.history_dim, H, W)
        
        # 计算时序注意力权重
        attention_weights = []
        for t in range(T):
            att_weight = self.temporal_attention(history_feat[:, t])  # (B, 1, H, W)
            attention_weights.append(att_weight)
        
        attention_weights = torch.stack(attention_weights, dim=1)  # (B, T, 1, H, W)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权融合
        weighted_feat = history_feat * attention_weights
        fused_feat = torch.sum(weighted_feat, dim=1)  # (B, history_dim, H, W)
        
        # 降维到hidden_dim
        fused_feat = nn.Conv2d(self.history_dim, self.hidden_dim, 1).to(history_feat.device)(fused_feat)
        
        return fused_feat

    def process_history_features_simple(self, history_blind_maps):
        """简单的历史特征处理方案"""
        B, T, H, W = history_blind_maps.shape
        
        # Process each history map
        history_feat = self.history_net(history_blind_maps.view(-1, 1, H, W))
        history_feat = history_feat.view(B, T, self.history_dim, H, W)
        
        # 简单平均融合
        fused_feat = torch.mean(history_feat, dim=1)  # (B, history_dim, H, W)
        
        # 降维到hidden_dim
        fused_feat = nn.Conv2d(self.history_dim, self.hidden_dim, 1).to(history_feat.device)(fused_feat)
        
        return fused_feat
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def create_location_feature_binary(self, ego_pillar_loc_in_agent, H, W):
        # 创建位置特征图
        B = len(ego_pillar_loc_in_agent)
        loc_features = torch.zeros(
            (B, 1, H, W), device=ego_pillar_loc_in_agent[0].device
        )

        for i, loc in enumerate(ego_pillar_loc_in_agent):
            # 将归一化坐标转换为图像坐标
            x = ((loc[0] + 1) * W / 2).long().clamp(0, W - 1)
            y = ((loc[1] + 1) * H / 2).long().clamp(0, H - 1)
            # print('ego location in bev map:', x, y)
            loc_features[i, 0, y, x] = 1.0

        return loc_features

    def create_location_feature_enhanced(self, ego_pillar_loc_in_agent, H, W):
        B = len(ego_pillar_loc_in_agent)
        loc_features = torch.zeros(
            (B, 2, H, W), device=ego_pillar_loc_in_agent[0].device
        )

        for i, loc in enumerate(ego_pillar_loc_in_agent):
            # 生成网格坐标
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, H, device=loc.device),
                torch.linspace(-1, 1, W, device=loc.device),
            )

            # 计算到ego的距离和角度
            dist = torch.sqrt((x_grid - loc[0]) ** 2 + (y_grid - loc[1]) ** 2)
            angle = torch.atan2(y_grid - loc[1], x_grid - loc[0])

            # 归一化并存储
            loc_features[i, 0] = torch.exp(-dist / 0.5)  # 距离特征
            loc_features[i, 1] = (angle + torch.pi) / (2 * torch.pi)  # 角度特征

        return loc_features

    def create_ray_integrated_encoding_without_feature(
        self, ego_pillar_loc_in_agent, H, W, pixel_size =None
    ):
        """
        创建射线积分位置编码 (RIPE) RIPE: Ray-informed Positional Encoding
        的简化版本，仅对位置进行编码，不融合特征。

        参数:
        - ego_pillar_loc_in_agent: ego车辆在agent坐标系中的位置列表
        - H, W: 特征图高度和宽度

        返回:
        - 射线积分位置编码特征 (B, self.ripe_dim, H, W)
        """
        B = len(ego_pillar_loc_in_agent)
        device = ego_pillar_loc_in_agent[0].device

        # 初始化射线编码特征
        ripe_features = torch.zeros((B, self.ripe_dim, H, W), device=device)
        
        
        if pixel_size is not None:
             # 计算BEV范围的物理尺寸（米）
            x_range = W * pixel_size  # BEV的x轴物理范围
            y_range = H * pixel_size  # BEV的y轴物理范围
            # 构建物理坐标网格（以中心为原点，BEV坐标）
            x_coords = torch.linspace(-x_range / 2, x_range / 2, W, device=device)
            y_coords = torch.linspace(-y_range / 2, y_range / 2, H, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")  # shape (H, W)
            diag_len = torch.sqrt(torch.tensor(x_range**2 + y_range**2, device=device))
            physical_decay = diag_len / 5  # 控制衰减速度
            for i, loc in enumerate(ego_pillar_loc_in_agent):
                # 使用原始坐标，不强制裁剪到可视范围内
                ego_x_norm = loc[0]  # 原始归一化坐标 [-1, 1]
                ego_y_norm = loc[1]  # 原始归一化坐标 [-1, 1]
                
                # 转换到物理坐标
                ego_x_physical = ego_x_norm * x_range / 2  # 归一化坐标转换到物理坐标
                ego_y_physical = ego_y_norm * y_range / 2  # 归一化坐标转换到物理坐标
                
                dx = x_grid - ego_x_physical
                dy = y_grid - ego_y_physical
                distances = torch.sqrt(dx**2 + dy**2)
                # 对于距离为0的点(ego位置)，设置一个很小的值避免除零
                distances = torch.clamp(distances, min=1e-5)
                # 对于ripe_dim指定的每个编码维度生成特征
                for c in range(self.ripe_dim):
                    if c == 0:
                        # 使用物理距离作为第一个编码维度，使用衰减函数
                        # 调整衰减因子以适应物理距离
                        # 设置衰减因子为BEV对角线长度的1/5
                        # 指数衰减距离
                        ripe_features[i, c] = torch.exp(-distances / physical_decay)
                    elif c == 1:
                        angles = torch.atan2(dy, dx)  # [-pi, pi]
                        # 归一化角度到 [0, 1]
                        ripe_features[i, c] = (angles + torch.pi) / (2 * torch.pi)
                    else:
                        # 对于额外的维度，使用基于物理距离的编码
                        # 使用不同频率的正弦函数
                        # 频率随着通道索引增加而增加，但基于实际物理距离
                        # 高频正弦编码：可模拟 positional encoding 风格
                        freq = 0.1 * (c - 1)
                        ripe_features[i, c] = torch.sin(distances * freq)
        else:
            for i, loc in enumerate(ego_pillar_loc_in_agent):
                # 将归一化坐标转换为图像坐标
                ego_x = ((loc[0] + 1) * W / 2).clamp(0, W - 1)
                ego_y = ((loc[1] + 1) * H / 2).clamp(0, H - 1)

                # 生成坐标网格
                y_indices, x_indices = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                )
                y_indices, x_indices = y_indices.float(), x_indices.float()

                # 计算每个点到ego的相对位置和距离
                dx = x_indices - ego_x
                dy = y_indices - ego_y
                distances = torch.sqrt(dx**2 + dy**2)

                # 对于距离为0的点(ego位置)，设置一个很小的值避免除零
                distances = torch.clamp(distances, min=1e-5)

                # 计算方向向量 (归一化)
                dir_x = dx / distances
                dir_y = dy / distances

                # 对位置进行编码
                for c in range(self.ripe_dim):
                    if c == 0:
                        # 使用距离作为第一个编码维度
                        ripe_features[i, c] = torch.exp(-distances / self.decay_lambda)
                    elif c == 1:
                        # 使用角度作为第二个编码维度
                        angles = torch.atan2(dy, dx)  # [-pi, pi]
                        ripe_features[i, c] = (angles + torch.pi) / (2 * torch.pi)
                    else:
                        # 对于额外的维度，可以使用其他函数对位置进行编码
                        ripe_features[i, c] = torch.sin(distances / (c + 1)) * torch.cos(angles / (c + 1))

        return ripe_features

    def get_pillar_loc(self, B, record_len, pairwise_t_matrix):
        """
        计算每个ego车辆在ego、infra坐标下的位置
        在传递给 BlindMap 之前，pairwise_t_matrix 中的平移部分已经被归一化到 [-1, 1] 范围。
        """
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
                # 检查坐标是否在有效范围内
                valid_range = 1.0  # 归一化坐标的有效范围
                if abs(tx) > valid_range or abs(ty) > valid_range:
                    # # 方案1：将超出范围的坐标裁剪到边界
                    # tx = torch.clamp(tx, -valid_range, valid_range)
                    # ty = torch.clamp(ty, -valid_range, valid_range)
                    
                    # 方案2：为超出范围的位置设置特殊标记（比如设为None或特殊值）
                    # tx = None
                    # ty = None
                    
                    # 方案3：记录是否超出范围，用于后续处理
                    out_of_range = True
                    # print('ego location out of infra lidar range:', tx, ty)
                else:
                    out_of_range = False
                
                ego_pillar_loc_in_agent.append(
                    torch.tensor([tx, ty], dtype=t_matrix.dtype, device=t_matrix.device)
                )
                # 将归一化坐标转换为物理坐标
                # tx_physical = tx * (self.downsample_rate * self.discrete_ratio * W) / 2
                # ty_physical = ty * (self.downsample_rate * self.discrete_ratio * H) / 2
                # loc_in_agents.append([tx_physical, ty_physical])
        return ego_pillar_loc_in_agent

    def forward(self, x, record_len, pairwise_t_matrix, pixel_size=None, history_blind_maps=None):
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
        history_blind_maps : torch.Tensor, optional
            History blind maps for each cav, shape: (B, T, 1, H, W)
            where T is the number of history frames.
        Returns
        -------
        Fused feature.
        """
        if len(history_blind_maps.shape) == 5:  # (B, T, 1, H, W)
            history_blind_maps = history_blind_maps.squeeze(2)
        _, C, H, W = x.shape
        B, L, _, _, _ = pairwise_t_matrix.shape
        # 1. Get ego locations
        ego_pillar_loc_in_agent = self.get_pillar_loc(B, record_len, pairwise_t_matrix)

        # 2. Create location features
        if self.use_ripe:
            ego_loc_features = self.create_ray_integrated_encoding_without_feature(
                ego_pillar_loc_in_agent, H, W, pixel_size
            )
        else:
            ego_loc_features = self.create_location_feature_binary(
                ego_pillar_loc_in_agent, H, W
            )
        # 3. Process history features if available
        if self.use_history and history_blind_maps is not None:
            history_blind_maps = history_blind_maps.float()
            # 选择一种历史特征处理方案
            # history_feat = self.process_history_features_3d(history_blind_maps)
            # 或者使用: history_feat = self.process_history_features_attention(history_blind_maps)
            history_feat = self.process_history_features_simple(history_blind_maps)
            # print(f"History features shape: {history_feat.shape}")
            # torch.Size([B, 64, 256, 256])
            # Concatenate all features
            # 为ego车辆插入零特征
            expanded_history_feat = []
            history_idx = 0  # 用于索引非ego的历史特征
            
            for b in range(B):
                N_cav = record_len[b]
                batch_features = []
                
                for cav_idx in range(N_cav):
                    if cav_idx == 0:  # ego车辆 (每个batch的第一个)
                        # 为ego插入零特征
                        zero_feat = torch.zeros(
                            (1, self.hidden_dim, H, W), 
                            device=history_feat.device, 
                            dtype=history_feat.dtype
                        )
                        batch_features.append(zero_feat)
                    else:  # 非ego车辆
                        # 使用实际的历史特征
                        batch_features.append(history_feat[history_idx:history_idx+1])
                        history_idx += 1
                
                # 拼接当前batch的所有特征
                batch_feat = torch.cat(batch_features, dim=0)  # (N_cav, hidden_dim, H, W)
                expanded_history_feat.append(batch_feat)
            
            # 拼接所有batch的特征
            expanded_history_feat = torch.cat(expanded_history_feat, dim=0)  # (sum(n_cav), hidden_dim, H, W)
            # print(x.shape, ego_loc_features.shape, expanded_history_feat.shape)
            # torch.Size([8, 64, 256, 256]) torch.Size([8, 2, 256, 256]) torch.Size([6, 64, 256, 256])
            fused_features = torch.cat([x, ego_loc_features, expanded_history_feat], dim=1)
        else:
            # Without history features
            fused_features = torch.cat([x, ego_loc_features], dim=1)
            # print(f"Fused features shape without history: {fused_features.shape}")
        # 4. 通过网络预测盲区
        out = self.relu(self.conv1(fused_features))
        out = self.relu(self.conv2(out))
        blind_map = self.sigmoid(self.conv3(out))  # (sum(n_cav), 1, H, W)

        return blind_map
