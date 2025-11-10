import torch
import torch.nn as nn
import torch.nn.functional as F

class PostFusionBlindMap(nn.Module):
    """后融合盲区预测模块 - 先预测当前帧，再与历史信息融合"""
    
    def __init__(self, model_cfg, in_channels=64, bev_H=128, bev_W=256):
        super(PostFusionBlindMap, self).__init__()
        print('----------Using PostFusionBlindMap------------')
        # 基本参数
        self.use_ripe = model_cfg.get('use_ripe', True)
        self.ripe_dim = model_cfg.get('ripe_dim', 8)
        self.decay_lambda = model_cfg.get('decay_lambda', 10.0)
        self.hidden_dim = model_cfg.get('hidden_dim', 64)
        self.use_history = model_cfg.get('use_history', False)
        self.history_num = model_cfg.get('history_num', 3)
        print('history_num:', self.history_num)
        # 当前帧盲区预测网络（保持原有结构）
        base_channels = in_channels + (self.ripe_dim if self.use_ripe else 1)
        self.current_predictor = nn.Sequential(
            nn.Conv2d(base_channels, self.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 历史信息融合策略选择
        fusion_strategy = model_cfg.get('history_fusion_strategy', 'weighted_average')
        
        if self.use_history:
            if fusion_strategy == 'weighted_average':
                self.history_fusion = WeightedAverageFusion(self.history_num)
            elif fusion_strategy == 'attention_guidance':
                self.history_fusion = AttentionGuidanceFusion(self.history_num)
            elif fusion_strategy == 'confidence_modulation':
                self.history_fusion = ConfidenceModulationFusion(self.history_num)
            elif fusion_strategy == 'adaptive_selection':
                self.history_fusion = AdaptiveSelectionFusion(self.history_num)
            else:
                self.history_fusion = SimpleAverageFusion(self.history_num)
    
    def forward(self, x, record_len, pairwise_t_matrix, pixel_size=None, history_blind_maps=None):
        """
        Args:
            x: 当前帧特征 (sum(n_cav), C, H, W)
            record_len: 每个batch中agent数量的列表 [n_cav_batch0, n_cav_batch1, ...]
            history_blind_maps: 历史盲区图 (sum(n_cav-1), T, H, W) - 不包括ego的历史信息
        """
        if history_blind_maps is not None and len(history_blind_maps.shape) == 5:
            history_blind_maps = history_blind_maps.squeeze(2)
        
        _, C, H, W = x.shape
        B, L, _, _, _ = pairwise_t_matrix.shape
        
        # 1. 获取位置编码
        ego_pillar_loc_in_agent = self.get_pillar_loc(B, record_len, pairwise_t_matrix)
        
        if self.use_ripe:
            ego_loc_features = self.create_ray_integrated_encoding_without_feature(
                ego_pillar_loc_in_agent, H, W, pixel_size
            )
        else:
            ego_loc_features = self.create_location_feature_binary(
                ego_pillar_loc_in_agent, H, W
            )
        
        # 2. 预测当前帧盲区
        current_features = torch.cat([x, ego_loc_features], dim=1)
        current_blind_map = self.current_predictor(current_features)  # (sum(n_cav), 1, H, W)
        
        # 3. 如果有历史信息，对非ego agent进行后融合
        if self.use_history and history_blind_maps is not None:
            history_blind_maps = history_blind_maps.float()
            final_blind_map = self._fuse_with_history(current_blind_map, history_blind_maps, record_len)
        else:
            final_blind_map = current_blind_map
        
        return final_blind_map
    def _fuse_with_history(self, current_blind_map, history_blind_maps, record_len):
        """
        将当前预测与历史信息融合
        
        Args:
            current_blind_map: (sum(n_cav), 1, H, W) - 所有agent的当前预测
            history_blind_maps: (sum(n_cav-1), T, H, W) - 非ego agent的历史信息
            record_len: 每个batch中agent数量的列表
        
        Returns:
            fused_blind_map: (sum(n_cav), 1, H, W) - 融合后的盲区图
        """
        fused_maps = []
        current_idx = 0  # 当前帧索引
        history_idx = 0  # 历史帧索引
        
        for batch_size in record_len:
            # 当前batch的所有agent预测
            batch_current = current_blind_map[current_idx:current_idx + batch_size]  # (n_cav, 1, H, W)
            
            # ego agent: 直接使用当前预测
            ego_current = batch_current[0:1]  # (1, 1, H, W)
            fused_maps.append(ego_current)
            
            # 非ego agent: 与历史信息融合
            if batch_size > 1:  # 如果有非ego agent
                non_ego_current = batch_current[1:]  # (n_cav-1, 1, H, W)
                non_ego_history = history_blind_maps[history_idx:history_idx + batch_size - 1]  # (n_cav-1, T, H, W)
                
                # 对每个非ego agent分别融合
                for i in range(batch_size - 1):
                    agent_current = non_ego_current[i:i+1]  # (1, 1, H, W)
                    agent_history = non_ego_history[i:i+1]  # (1, T, H, W)
                    
                    # 使用融合策略
                    fused_agent = self.history_fusion(agent_current, agent_history)  # (1, 1, H, W)
                    fused_maps.append(fused_agent)
                
                history_idx += batch_size - 1
            
            current_idx += batch_size
        
        return torch.cat(fused_maps, dim=0)  # (sum(n_cav), 1, H, W)
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
class SimpleAverageFusion(nn.Module):
    """最简单的平均融合"""
    
    def __init__(self, history_num=3):
        super().__init__()
        self.history_num = history_num
        # 固定的时间衰减权重
        weights = torch.exp(-torch.arange(history_num, dtype=torch.float32) * 0.3)
        weights = weights / weights.sum()
        self.register_buffer('time_weights', weights)
    
    def forward(self, current_blind_map, history_blind_maps):
        """
        Args:
            current_blind_map: (1, 1, H, W) - 单个agent的当前帧预测
            history_blind_maps: (1, T, H, W) - 单个agent的历史帧预测
        """
        _, T, H, W = history_blind_maps.shape
        
        # 计算历史信息的加权平均
        weights = self.time_weights.view(1, T, 1, 1)
        history_avg = torch.sum(history_blind_maps * weights, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # 当前帧与历史平均的融合 (0.7当前 + 0.3历史)
        fused = 0.7 * current_blind_map + 0.3 * history_avg
        
        return fused


class WeightedAverageFusion(nn.Module):
    """学习权重的加权平均融合"""
    
    def __init__(self, history_num=3):
        super().__init__()
        self.history_num = history_num
        
        # 学习当前帧和历史帧的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.7))  # 当前帧权重
        
        # 历史帧的时序权重
        self.temporal_weights = nn.Parameter(
            torch.exp(-torch.arange(self.history_num, dtype=torch.float32) * 0.3)
        )
    
    def forward(self, current_blind_map, history_blind_maps):
        _, T, H, W = history_blind_maps.shape
        if T < self.history_num:
            pad_shape = (history_blind_maps.shape[0], self.history_num - T, H, W)
            pad = torch.zeros(pad_shape, device=history_blind_maps.device, dtype=history_blind_maps.dtype)
            history_blind_maps = torch.cat([history_blind_maps, pad], dim=1)
        # 归一化时序权重
        T =  self.history_num
        normalized_weights = F.softmax(self.temporal_weights, dim=0)
        weights = normalized_weights.view(1, T, 1, 1)
        
        # 历史信息加权平均
        history_weighted = torch.sum(history_blind_maps * weights, dim=1, keepdim=True)
        
        # 当前帧与历史的自适应融合
        current_weight = torch.sigmoid(self.fusion_weight)
        history_weight = 1 - current_weight
        
        fused = current_weight * current_blind_map + history_weight * history_weighted
        
        return fused


class AttentionGuidanceFusion(nn.Module):
    """基于注意力的引导融合"""
    
    def __init__(self, history_num=3):
        super().__init__()
        self.history_num = history_num
        
        # 轻量级注意力网络
        self.attention_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, current_blind_map, history_blind_maps):
        _, T, H, W = history_blind_maps.shape
        
        # 计算历史信息的简单平均
        history_avg = torch.mean(history_blind_maps, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # 基于当前预测生成注意力图
        attention_map = self.attention_net(current_blind_map)  # (1, 1, H, W)
        
        # 使用注意力引导历史信息的影响
        guided_history = history_avg * attention_map
        
        # 自适应融合
        fused = current_blind_map + 0.3 * guided_history
        fused = torch.clamp(fused, 0, 1)  # 保持在[0,1]范围
        
        return fused


class ConfidenceModulationFusion(nn.Module):
    """基于置信度调制的融合"""
    
    def __init__(self, history_num=3):
        super().__init__()
        self.history_num = history_num
    
    def forward(self, current_blind_map, history_blind_maps):
        _, T, H, W = history_blind_maps.shape
        
        # 计算当前预测的置信度（基于预测值的分布）
        current_confidence = self._compute_confidence(current_blind_map)  # (1, 1, H, W)
        
        # 历史信息的一致性
        history_consistency = self._compute_consistency(history_blind_maps)  # (1, 1, H, W)
        
        # 基于置信度调制融合权重
        # 高置信度区域更依赖当前预测，低置信度区域参考历史信息
        current_weight = current_confidence
        history_weight = (1 - current_confidence) * history_consistency
        
        # 历史平均
        history_avg = torch.mean(history_blind_maps, dim=1, keepdim=True)
        
        # 置信度调制融合
        fused = current_weight * current_blind_map + history_weight * history_avg
        
        return fused
    
    def _compute_confidence(self, blind_map):
        """计算预测置信度"""
        # 基于预测值的确定性 (接近0或1的值置信度高)
        confidence = 1 - 2 * torch.abs(blind_map - 0.5)
        return confidence
    
    def _compute_consistency(self, history_blind_maps):
        """计算历史信息的一致性"""
        _, T, H, W = history_blind_maps.shape
        
        # 计算历史帧间的方差
        history_mean = torch.mean(history_blind_maps, dim=1, keepdim=True)
        history_var = torch.mean((history_blind_maps - history_mean)**2, dim=1, keepdim=True)
        
        # 一致性 = 1 - 方差 (方差小说明一致性高)
        consistency = torch.exp(-history_var * 5)
        
        return consistency


class AdaptiveSelectionFusion(nn.Module):
    """自适应选择融合 - 动态决定是否使用历史信息"""
    
    def __init__(self, history_num=3):
        super().__init__()
        self.history_num = history_num
        
        # 决策网络 - 判断是否需要历史信息
        self.decision_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # 降维到4x4
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_blind_map, history_blind_maps):
        _, T, H, W = history_blind_maps.shape
        
        # 决策是否使用历史信息
        use_history_prob = self.decision_net(current_blind_map)  # (1, 1)
        use_history_prob = use_history_prob.view(1, 1, 1, 1)
        
        # 历史信息平均
        history_avg = torch.mean(history_blind_maps, dim=1, keepdim=True)
        
        # 自适应选择
        fused = (1 - use_history_prob) * current_blind_map + use_history_prob * (
            0.6 * current_blind_map + 0.4 * history_avg
        )
        
        return fused
