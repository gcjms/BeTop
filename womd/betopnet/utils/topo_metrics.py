'''
Topology Metrics for BeTop
拓扑预测准确率计算

支持二分类和三分类指标：
- 二分类：有冲突/无冲突
- 三分类：让/无关/超
'''

import torch
import torch.nn.functional as F


def compute_topo_accuracy_binary(pred, gt, mask):
    """
    计算二分类拓扑预测准确率（有冲突/无冲突）
    
    Args:
        pred: [B, 1, N, 1] 模型预测 logits
        gt: [B, 1, N, T] GT 标签（0: 无冲突, 非0: 有冲突）
        mask: [B, 1, N] 有效性掩码
    
    Returns:
        dict: precision, recall, f1, accuracy
    """
    # 二分类处理
    pred_binary = (torch.sigmoid(pred) > 0.5).float().squeeze(-1)  # [B, 1, N]
    gt_binary = (gt[..., 0] != 0).float()  # [B, 1, N] 取第一个时间步
    
    mask = mask.float()
    
    TP = ((pred_binary == 1) & (gt_binary == 1) & (mask == 1)).sum()
    FP = ((pred_binary == 1) & (gt_binary == 0) & (mask == 1)).sum()
    FN = ((pred_binary == 0) & (gt_binary == 1) & (mask == 1)).sum()
    TN = ((pred_binary == 0) & (gt_binary == 0) & (mask == 1)).sum()
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    
    return {
        'topo_binary_precision': precision.item(),
        'topo_binary_recall': recall.item(),
        'topo_binary_f1': f1.item(),
        'topo_binary_accuracy': accuracy.item(),
    }


def compute_topo_accuracy_3class(pred, gt, mask):
    """
    计算三分类拓扑预测准确率（让/无关/超）
    
    Args:
        pred: [B, S, N, T, 3] 模型预测 logits（3分类）
        gt: [B, S, N, T] GT 标签（-1: 让, 0: 无关, +1: 超）
        mask: [B, S, N] 有效性掩码
    
    Returns:
        dict: 各类别指标和整体准确率
    """
    B, S, N, T, C = pred.shape
    
    # 预测类别：argmax 后映射回 {-1, 0, +1}
    pred_class = pred.argmax(dim=-1) - 1  # [B, S, N, T], 值域 {-1, 0, +1}
    gt_class = gt.long()  # [B, S, N, T]
    
    # 扩展 mask 到时间维度
    mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, T).float()
    total_valid = mask_exp.sum() + 1e-6
    
    # 整体准确率
    correct = ((pred_class == gt_class) & (mask_exp == 1)).sum()
    accuracy = correct / total_valid
    
    # 各类别准确率
    metrics = {'topo_3class_accuracy': accuracy.item()}
    
    class_names = {-1: 'yield', 0: 'none', 1: 'pass'}
    for cls_val, cls_name in class_names.items():
        cls_mask = (gt_class == cls_val) & (mask_exp == 1)
        cls_total = cls_mask.sum() + 1e-6
        cls_correct = ((pred_class == cls_val) & cls_mask).sum()
        
        # Precision: 预测为该类别的样本中，真正是该类别的比例
        pred_cls_mask = (pred_class == cls_val) & (mask_exp == 1)
        precision = ((gt_class == cls_val) & pred_cls_mask).sum() / (pred_cls_mask.sum() + 1e-6)
        
        # Recall: 真正是该类别的样本中，被正确预测的比例
        recall = cls_correct / cls_total
        
        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics[f'topo_{cls_name}_precision'] = precision.item()
        metrics[f'topo_{cls_name}_recall'] = recall.item()
        metrics[f'topo_{cls_name}_f1'] = f1.item()
    
    return metrics


def compute_topo_metrics(pred, gt, mask, num_classes=1):
    """
    统一接口：根据输出维度自动选择二分类或三分类指标
    
    Args:
        pred: 
            - 二分类: [B, 1, N, multi_step] 
            - 三分类: [B, 1, N, multi_step, 3]
        gt: [B, 1, N, multi_step] GT 标签
        mask: [B, 1, N] 有效性掩码
        num_classes: 分类数（1=二分类，3=三分类）
    
    Returns:
        dict: 准确率指标
    """
    if num_classes == 3 or len(pred.shape) == 5:
        return compute_topo_accuracy_3class(pred, gt, mask)
    else:
        return compute_topo_accuracy_binary(pred, gt, mask)


def topo_loss_3class(pred, gt, mask, class_weights=None, gamma=2.0, alpha=0.25):
    """
    三分类拓扑损失函数（Focal Loss 版本）
    
    Focal Loss: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    - 对易分类样本降低权重
    - 聚焦于难分类样本
    
    Args:
        pred: [B, S, N, T, C] 模型预测 logits (S=1, T=multi_step, C=3)
        gt: [B, S, N, T] GT 标签（-1: 让, 0: 无关, +1: 超）
        mask: [B, S, N] 有效性掩码
        class_weights: [3] 类别权重（可选）
        gamma: Focal Loss 聚焦参数，越大越聚焦难样本（默认 2.0）
        alpha: 平衡因子（默认 0.25）
    
    Returns:
        loss: [B] 每个样本的损失
    """
    B, S, N, T, C = pred.shape
    
    # 将 GT 从 {-1, 0, +1} 映射到 {0, 1, 2}
    gt_class = (gt + 1).long()  # [B, S, N, T], 值域 {0, 1, 2}
    gt_class = gt_class.clamp(0, 2)  # 确保合法范围
    
    # 扩展 mask 到时间维度 [B, S, N] -> [B, S, N, T]
    mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, T)
    
    # Reshape for cross entropy: 合并所有维度
    pred_flat = pred.reshape(-1, C)  # [B*S*N*T, 3]
    gt_flat = gt_class.reshape(-1)   # [B*S*N*T]
    mask_flat = mask_exp.reshape(-1).float()  # [B*S*N*T]
    
    # 计算 CrossEntropy（不 reduce，得到每个样本的 loss）
    if class_weights is not None:
        class_weights = class_weights.to(pred.device)
        ce_loss = F.cross_entropy(pred_flat, gt_flat, weight=class_weights, reduction='none')
    else:
        ce_loss = F.cross_entropy(pred_flat, gt_flat, reduction='none')
    
    # 计算 Focal Loss 权重
    # p_t = exp(-ce_loss) = softmax(pred)[gt_class]
    pt = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Focal Loss = focal_weight * ce_loss
    loss = focal_weight * ce_loss
    
    # 应用 mask
    loss = loss * mask_flat
    
    # Reshape 回 [B, S*N*T] 并求和
    loss = loss.view(B, S * N * T)
    mask_sum = mask_exp.reshape(B, -1).sum(dim=-1).clamp(min=1)
    
    return loss.sum(dim=-1) / mask_sum

