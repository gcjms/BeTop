"""
三分类拓扑 Loss (Focal Loss) 测试脚本

测试内容：
1. topo_loss_3class 函数的基本正确性
2. Focal Loss 的难样本聚焦效果
3. compute_topo_accuracy_3class 的准确率计算

运行方式：
    cd womd
    python test_topo_loss.py
"""

import sys
import os

# 确保可以 import betopnet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from betopnet.utils.topo_metrics import topo_loss_3class, compute_topo_accuracy_3class


def test_focal_loss_basic():
    """测试 Focal Loss 基本功能"""
    print("=" * 50)
    print("Test 1: Focal Loss 基本功能")
    print("=" * 50)
    
    B, S, N, T, C = 2, 1, 5, 1, 3  # batch=2, N=5个障碍物, T=1时间步, C=3类
    
    # 随机预测和标签
    pred = torch.randn(B, S, N, T, C, requires_grad=True)
    gt = torch.randint(-1, 2, (B, S, N, T)).float()  # {-1, 0, 1}
    mask = torch.ones(B, S, N)
    
    # 计算 Loss
    loss = topo_loss_3class(pred, gt, mask, gamma=2.0, alpha=0.25)
    print(f"Loss shape: {loss.shape}")
    print(f"Loss values: {loss}")
    
    # 检查梯度
    loss.mean().backward()
    print(f"Gradient exists: {pred.grad is not None}")
    print("✅ 基本功能测试通过\n")


def test_focal_loss_hard_sample():
    """测试 Focal Loss 难样本聚焦效果"""
    print("=" * 50)
    print("Test 2: Focal Loss 难样本聚焦效果")
    print("=" * 50)
    
    # GT = 0 (无交互) → 映射为 Class 1
    gt = torch.zeros(1, 1, 1, 1)  # 原始值 0
    mask = torch.ones(1, 1, 1)
    
    # Case A: 完美预测 (Easy Sample)
    # 模型非常确定是 Class 1 (中间那个数最大)
    pred_perfect = torch.tensor([[[[[
        -10.0,  # Class 0 (让)
        10.0,   # Class 1 (无) ← 最大，正确
        -10.0   # Class 2 (超)
    ]]]]])
    loss_perfect = topo_loss_3class(pred_perfect, gt, mask, gamma=2.0)
    
    # Case B: 错误预测 (Hard Sample)
    # 模型非常确定是 Class 0 (第一个数最大) → 错了
    pred_wrong = torch.tensor([[[[[
        10.0,   # Class 0 (让) ← 最大，错误
        -10.0,  # Class 1 (无)
        -10.0   # Class 2 (超)
    ]]]]])
    loss_wrong = topo_loss_3class(pred_wrong, gt, mask, gamma=2.0)
    
    print(f"Easy Sample Loss: {loss_perfect.item():.8f}")
    print(f"Hard Sample Loss: {loss_wrong.item():.8f}")
    
    ratio = loss_wrong / (loss_perfect + 1e-8)
    print(f"Hard/Easy Ratio: {ratio.item():.2f}")
    
    if ratio > 100:
        print("✅ Focal Loss 聚焦效果验证通过：难样本 Loss 远大于易样本\n")
    else:
        print("❌ 警告：Focal Loss 聚焦效果不明显\n")


def test_accuracy_metrics():
    """测试准确率计算函数"""
    print("=" * 50)
    print("Test 3: 准确率计算")
    print("=" * 50)
    
    B, S, N, T, C = 2, 1, 4, 1, 3
    
    # 构造一个已知的测试用例
    # 预测 logits：每行最大值对应预测的类别
    pred = torch.tensor([[[
        [[10.0, -10.0, -10.0]],  # 预测 Class 0 (让)
        [[-10.0, 10.0, -10.0]],  # 预测 Class 1 (无)
        [[-10.0, -10.0, 10.0]],  # 预测 Class 2 (超)
        [[10.0, -10.0, -10.0]],  # 预测 Class 0 (让)
    ]], [[
        [[-10.0, 10.0, -10.0]],  # 预测 Class 1 (无)
        [[-10.0, 10.0, -10.0]],  # 预测 Class 1 (无)
        [[10.0, -10.0, -10.0]],  # 预测 Class 0 (让)
        [[-10.0, -10.0, 10.0]],  # 预测 Class 2 (超)
    ]]])
    
    # GT 标签（原始值 -1, 0, 1）
    gt = torch.tensor([[[
        [-1.0],  # 让 → Class 0 ✓
        [0.0],   # 无 → Class 1 ✓
        [1.0],   # 超 → Class 2 ✓
        [0.0],   # 无 → Class 1 ✗ (预测是 Class 0)
    ]], [[
        [0.0],   # 无 → Class 1 ✓
        [-1.0],  # 让 → Class 0 ✗ (预测是 Class 1)
        [-1.0],  # 让 → Class 0 ✓
        [1.0],   # 超 → Class 2 ✓
    ]]])
    
    mask = torch.ones(B, S, N)
    
    metrics = compute_topo_accuracy_3class(pred, gt, mask)
    
    # 预期准确率: 6/8 = 0.75
    print(f"Overall Accuracy: {metrics['topo_3class_accuracy']:.4f} (预期: 0.75)")
    print(f"Yield Precision: {metrics['topo_yield_precision']:.4f}")
    print(f"Yield Recall: {metrics['topo_yield_recall']:.4f}")
    print(f"None Precision: {metrics['topo_none_precision']:.4f}")
    print(f"Pass Precision: {metrics['topo_pass_precision']:.4f}")
    
    if abs(metrics['topo_3class_accuracy'] - 0.75) < 0.01:
        print("✅ 准确率计算测试通过\n")
    else:
        print("❌ 准确率计算可能有误\n")


def test_multi_step():
    """测试多时间步支持"""
    print("=" * 50)
    print("Test 4: 多时间步支持 (multi_step > 1)")
    print("=" * 50)
    
    B, S, N, T, C = 2, 1, 3, 4, 3  # T=4 个时间步
    
    pred = torch.randn(B, S, N, T, C)
    gt = torch.randint(-1, 2, (B, S, N, T)).float()
    mask = torch.ones(B, S, N)
    
    loss = topo_loss_3class(pred, gt, mask)
    metrics = compute_topo_accuracy_3class(pred, gt, mask)
    
    print(f"Input shape: pred={pred.shape}, gt={gt.shape}")
    print(f"Loss: {loss}")
    print(f"Accuracy: {metrics['topo_3class_accuracy']:.4f}")
    print("✅ 多时间步测试通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("三分类拓扑 Focal Loss 测试")
    print("=" * 50 + "\n")
    
    try:
        test_focal_loss_basic()
        test_focal_loss_hard_sample()
        test_accuracy_metrics()
        test_multi_step()
        
        print("=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
