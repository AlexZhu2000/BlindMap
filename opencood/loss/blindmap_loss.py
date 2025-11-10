import torch
import torch.nn as nn
import torch.nn.functional as F
focal_bceloss = True
dice_loss = True
class BlindMapLoss(nn.Module):
    def __init__(self, args):
        super(BlindMapLoss, self).__init__()
        self.loss_weight = args['blind_map_loss_weight']
        self.dice_smooth = 1.0  # 平滑因子，防止分母为零
        # Define downsample factor based on backbone configuration
        self.downsample_factor = 2  # ResNet reduces spatial dimensions by 2
    def focal_bce_loss(self, pred, target, gamma=2.0, alpha=0.25):
        """
        改进的Focal BCE损失
        结合BCE损失和Focal Loss的思想
        """
        # 避免数值不稳定
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # 标准BCE损失
        bce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Focal Loss权重
        pt = torch.exp(-bce_loss)
        focal_weight = (alpha * (1 - pt) ** gamma)
        
        return (focal_weight * bce_loss).mean()
    def dice_loss(self, pred, target, smooth=1.0):
        """
        计算Dice Loss
        
        Args:
            pred (tensor): 预测的概率图
            target (tensor): 目标二值图
            smooth (float): 平滑因子，防止分母为零
            
        Returns:
            dice_loss (tensor): Dice Loss值
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_coef = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice_coef
    def forward(self, pred_blind_maps, gt_blind_maps, record_len):
        """
        Compute supervision loss for blind maps using only the infra's prediction
        
        Args:
            pred_blind_maps (tensor): shape (sum(cav), 1, H/2, W/2)
            gt_blind_maps (tensor): shape (B, H, W)
            record_len (list): list of number of CAVs per batch
        
        Returns:
            loss (tensor): Blind map loss value
        """
        # Downsample ground truth to match prediction resolution
        gt_blind_maps = F.interpolate(
            gt_blind_maps.unsqueeze(1),  # Add channel dim: (B, 1, H, W)
            scale_factor=1/self.downsample_factor, 
            mode='nearest'
        )  # (B, 1, H/2, W/2)
         # 获取所有非ego智能体的预测
        non_ego_preds = []
        start_idx = 0
        for num_cav in record_len:
            # 跳过每个batch的第一个智能体(ego)
            non_ego_preds.append(pred_blind_maps[start_idx+1:start_idx+num_cav])
            start_idx += num_cav
        
        # 连接所有非ego预测
        pred_maps = torch.cat(non_ego_preds, dim=0)  # (sum(non_ego_cav), 1, H/2, W/2)
        # print('pred_maps shape:', pred_maps.shape)
        # print('gt_blind_maps shape:', gt_blind_maps.shape)
        # pred_maps shape: torch.Size([7, 1, 128, 256])
        # gt_blind_maps shape: torch.Size([7, 1, 128, 256])
        # 计算损失
        noise = torch.randn_like(gt_blind_maps) * 0.05
        noisy_gt = torch.clamp(gt_blind_maps + noise, 0, 1)
        total_loss = 0
        if dice_loss:
            total_loss += self.dice_loss(pred_maps, noisy_gt, self.dice_smooth)
        if focal_bceloss:
            total_loss += self.focal_bce_loss(pred_maps, noisy_gt)
        
        return total_loss * self.loss_weight
        # total_loss = 0
        # cumsum = torch.cumsum(torch.tensor(record_len), dim=0)
        # batch_size = len(record_len)
        
        # for b in range(batch_size):
        #     # Get infra's prediction (second agent) for this batch
        #     start_idx = 0 if b == 0 else cumsum[b-1]
        #     infra_idx = start_idx + 1  # index for infra (second agent)
            
        #     # Ensure infra index is valid
        #     if infra_idx >= pred_blind_maps.shape[0]:
        #         continue
                
        #     pred_map = pred_blind_maps[infra_idx:infra_idx+1]  # Shape: (1, 1, H, W)
            
        #     # Get corresponding ground truth
        #     gt_map = gt_blind_maps[b]  # Shape: (H, W)
        #     if dice_loss:
        #         # 计算Dice Loss
        #         loss = self.dice_loss(pred_map, gt_map, self.dice_smooth)
        #         total_loss += loss
        #     if focal_bceloss:
        #         gt_map = gt_map.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
                
        #         # Compute loss
        #         focal_bce_loss = self.focal_bce_loss(pred_map, gt_map)
        #         total_loss += focal_bce_loss
            
        # return total_loss * self.loss_weight / batch_size


# 测试代码示例 (可选，用于验证实现)
if __name__ == "__main__":
    # 创建模拟数据
    args = {'blind_map_loss_weight': 1.0}
    loss_fn = BlindMapLoss(args)
    
    # 假设有2个批次，每个批次有2个CAV
    pred_maps = torch.sigmoid(torch.randn(4, 1, 128, 128))  # 模拟sigmoid后的预测
    gt_maps = (torch.rand(2, 256, 256) > 0.8).float()  # 模拟二值真实标签
    record_len = [2, 2]  # 每个批次的CAV数量
    
    # 计算损失
    loss = loss_fn(pred_maps, gt_maps, record_len)
    print(f"Loss: {loss.item()}")