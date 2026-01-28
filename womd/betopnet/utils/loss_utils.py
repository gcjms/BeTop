'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''

import torch 
import torch.nn.functional as F

def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
    """
    ============================================================================
    Focal Loss - 用于处理类别不平衡的分类损失
    ============================================================================
    
    【核心思想】
    普通的交叉熵对所有样本一视同仁，但在目标检测/拓扑预测中，
    大部分样本是"简单的负样本"（比如没有交互的障碍物），
    少部分是"困难的正样本"（比如即将发生碰撞的车）。
    
    Focal Loss 通过 (1 - p_t)^γ 这个调制因子，让模型：
    - 对"已经预测得很准"的样本少关注（权重小）
    - 对"预测得很差"的样本多关注（权重大）
    
    【公式】
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    【参数说明】
    - inputs: 模型输出的 logits（未经 sigmoid）
    - targets: Ground Truth 标签 (0 或 1)
    - alpha: 正负样本的权重平衡因子 (默认 0.25，即正样本权重更小)
    - gamma: 聚焦因子 (默认 2，值越大越聚焦于难例)
    ============================================================================
    """
    # Step 1: 将 logits 转换为概率
    p = torch.sigmoid(inputs)
    
    # Step 2: 计算标准的二分类交叉熵损失
    # 这里用 with_logits 版本，内部会自动做 sigmoid，数值更稳定
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # Step 3: 计算 p_t（正确类别的预测概率）
    # 如果 target=1，p_t = p（模型预测为正的概率）
    # 如果 target=0，p_t = 1-p（模型预测为负的概率）
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Step 4: 应用调制因子 (1 - p_t)^γ
    # 当预测正确时（p_t 接近 1），(1-p_t)^γ 趋近于 0 → 权重很小
    # 当预测错误时（p_t 接近 0），(1-p_t)^γ 趋近于 1 → 权重保持
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Step 5: 应用 alpha 权重平衡正负样本
    # alpha_t = alpha（对正样本）或 (1-alpha)（对负样本）
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Step 6: 根据 reduction 参数决定如何聚合损失
    if reduction == "none":
        pass  # 保持原形状，不聚合
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def topo_loss(
    prediction, targets, valid_mask,
    top_k=False, top_k_ratio=1.):
    """
    ============================================================================
    BeTop 拓扑损失函数 (Topology Loss)
    ============================================================================
    
    【核心功能】
    计算拓扑预测（Braid 编码）与 Ground Truth 的 Focal Loss，
    并可选地使用 Top-K 策略只关注最难的样本。
    
    【输入参数】
    - prediction: [B, Src, Tgt, Step] 模型预测的拓扑分数
      - B: Batch Size
      - Src: 传入时通常是 1（因为在调用前已筛选出正样本模态） -> 64个intention里选出来和gt最像的那个了
      - Tgt: 目标实体数量（如 N_agents 个障碍物）
      - Step: 时间步数（如 8 个时间段的拓扑状态）
    - targets: [B, Src, Tgt, Step] Ground Truth 拓扑标签
    - valid_mask: [B, Src, Tgt] 有效性掩码（哪些障碍物是真实存在的）
    - top_k: 是否启用 Top-K 难例挖掘
    - top_k_ratio: 保留前 K% 的最难样本（默认 1.0 = 100%）
    
    【输出】
    - loss: [B] 每个样本的平均拓扑损失
    
    【Top-K 策略的意义】
    场景中可能有几十个障碍物，但只有少部分和自车有复杂交互。
    Top-K 策略让模型专注于那些"预测最差"的拓扑关系，
    忽略那些已经预测得很好的简单情况（如远处静止的车）。
    ============================================================================
    """
    # 获取各维度大小
    b, s, t, step = prediction.shape  # [Batch, Src, Tgt, TimeSteps]
    targets = targets.float()

    # Step 1: 计算逐元素的 Focal Loss
    # 输出形状: [B, S, T, Step]
    loss = focal_loss(
        prediction,
        targets,
        reduction='none',  # 保持所有维度，不聚合
    )

    # Step 2: 应用有效性掩码
    # valid_mask: [B, S, T] -> [B, S, T, 1] 扩展后与 loss 相乘
    # 无效的障碍物（padding）的 loss 置零
    loss = loss * valid_mask[..., None]
    
    # Step 3: reshape 以便进行 Top-K 排序
    # [B, S, T, Step] -> [B, S*T, St ep]
    loss = loss.view(b, s*t, step)
    valid_mask = valid_mask.view(b, s*t)
    
    # Step 4: Top-K 难例挖掘（可选）
    if top_k:
        # 只惩罚 loss 最大的前 K 个样本（最难的样本）
        k = int(top_k_ratio * loss.shape[1])  # 例如 25% 的样本
        # 按 loss 降序排序，取前 K 个
        loss, _ = torch.sort(loss, dim=1, descending=True)
        loss = loss[:, :k]  # [B, K, Step]
    
    # Step 5: 计算有效样本数量（用于归一化）
    mask = torch.sum(valid_mask, dim=-1)  # [B] 每个样本有多少有效实体
    mask = mask + (mask == 0).float()  # 防止除零
    
    # Step 6: 计算最终损失
    # loss.mean(-1): 对时间步取平均 -> [B, S*T] 或 [B, K]
    # sum(dim=1): 对所有实体求和 -> [B]
    # / mask: 归一化
    # 归一化 = 总 loss / 有效障碍物数量  = "障碍物平均贡献的 loss"
    return torch.sum(loss.mean(-1), dim=1) / mask