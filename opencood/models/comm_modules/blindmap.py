import torch
import torch.nn as nn
import numpy as np

C = 64


class BlindMap(nn.Module):
    def __init__(self, args, bev_channels=C):
        super(BlindMap, self).__init__()
        # 特征提取和融合网络
        print(args)
        self.ripe_dim =  args.get("ripe_dim", 2)  # 默认为2维射线编码
        self.use_ripe = args.get("use_ripe", True)  # 是否使用射线积分编码
        self.decay_lambda = args.get("decay_lambda", 0.5)  # 衰减因子
        self.batch_norm = args.get("batch_norm", False)  # 是否使用批归一化
        print("BlindMap: use_ripe:", self.use_ripe)
        print("BlindMap: ripe_dim:", self.ripe_dim)
        # 根据是否使用RIPE调整输入通道数
        input_channels = bev_channels + (self.ripe_dim if self.use_ripe else 1)
        # print('input_channels', input_channels)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)
        if self.batch_norm:
            self.conv1 = nn.Sequential(self.conv1, nn.BatchNorm2d(64))
            self.conv2 = nn.Sequential(self.conv2, nn.BatchNorm2d(32))
            self.conv3 = nn.Sequential(self.conv3, nn.BatchNorm2d(1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, x, record_len, pairwise_t_matrix, pixel_size=None):
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

        # 根据配置选择位置编码方法
        if self.use_ripe:
            ego_loc_features = self.create_ray_integrated_encoding_without_feature(
                ego_pillar_loc_in_agent, H, W, pixel_size
            )
        else:
            ego_loc_features = self.create_location_feature_binary(
                ego_pillar_loc_in_agent, H, W
            )
        # ego_loc_features = self.create_location_feature_enhanced(ego_pillar_loc_in_agent, H, W)

        # 3. 特征融合
        fused_features = torch.cat(
            [x, ego_loc_features], dim=1
        )  # (sum(n_cav), C+ripe_dim, H, W)
        # 4. 通过网络预测盲区
        out = self.relu(self.conv1(fused_features))
        out = self.relu(self.conv2(out))
        blind_map = self.sigmoid(self.conv3(out))  # (sum(n_cav), 1, H, W)

        return blind_map
