'''
Intersection Topology Detection for BeTop
路口冲突拓扑检测模块

检测 ego 与其他 agent 在路口的交互优先级：
- +1: ego 先到冲突点（抢行）
- -1: agent 先到冲突点（让行）  
- 0: 无冲突（轨迹不相交）
'''

import torch
import torch.nn.functional as F

# 从 topo_utils 导入线段相交检测函数
from betopnet.utils.topo_utils import segments_intersect


# ============================================================================
# 路口冲突检测核心函数
# ============================================================================

def find_trajectory_intersections(
    ego_traj,       # [B, T, 2] ego 轨迹
    agent_trajs,    # [B, N, T, 2] agent 轨迹
    ego_mask,       # [B, T] ego 有效 mask
    agent_mask,     # [B, N, T] agent 有效 mask
):
    """
    检测 ego 轨迹与每个 agent 轨迹是否有交叉点
    
    Returns:
        has_intersection: [B, N] 是否存在交叉
        intersection_t_ego: [B, N] ego 到达交叉点的时间索引
        intersection_t_agent: [B, N] agent 到达交叉点的时间索引
    """
    B, T, _ = ego_traj.shape
    N = agent_trajs.shape[1]
    device = ego_traj.device
    
    # 初始化结果
    has_intersection = torch.zeros(B, N, device=device, dtype=torch.bool)
    intersection_t_ego = torch.zeros(B, N, device=device, dtype=torch.float)
    intersection_t_agent = torch.zeros(B, N, device=device, dtype=torch.float)
    
    # 构建 ego 轨迹的线段: [B, T-1, 2] start/end
    ego_seg_start = ego_traj[:, :-1, :]  # [B, T-1, 2]
    ego_seg_end = ego_traj[:, 1:, :]     # [B, T-1, 2]
    ego_seg_mask = ego_mask[:, :-1] & ego_mask[:, 1:]  # [B, T-1]
    
    # 对每个 agent 检测交叉
    for n in range(N):
        # Agent n 的轨迹线段
        agent_seg_start = agent_trajs[:, n, :-1, :]  # [B, T-1, 2]
        agent_seg_end = agent_trajs[:, n, 1:, :]     # [B, T-1, 2]
        agent_seg_mask = agent_mask[:, n, :-1] & agent_mask[:, n, 1:]  # [B, T-1]
        
        # 检测所有 ego 线段 vs 所有 agent 线段的交叉
        # 扩展维度: ego [B, T-1, 1, 2], agent [B, 1, T-1, 2]
        ego_start_exp = ego_seg_start[:, :, None, :]  # [B, T-1, 1, 2]
        ego_end_exp = ego_seg_end[:, :, None, :]
        agent_start_exp = agent_seg_start[:, None, :, :]  # [B, 1, T-1, 2]
        agent_end_exp = agent_seg_end[:, None, :, :]
        
        # 检测交叉: [B, T-1, T-1]
        intersect_matrix = segments_intersect(
            ego_start_exp, ego_end_exp,
            agent_start_exp, agent_end_exp
        )
        
        # 应用 mask
        valid_mask = ego_seg_mask[:, :, None] & agent_seg_mask[:, None, :]  # [B, T-1, T-1]
        intersect_matrix = intersect_matrix & valid_mask
        
        # 是否存在任何交叉
        batch_has_inter = torch.any(intersect_matrix.view(B, -1), dim=-1)  # [B]
        has_intersection[:, n] = batch_has_inter
        
        # 找到第一个交叉点的时间索引
        for b in range(B):
            if batch_has_inter[b]:
                # 找到第一个交叉的 (ego_t, agent_t) 对
                inter_indices = torch.nonzero(intersect_matrix[b], as_tuple=False)
                if len(inter_indices) > 0:
                    # 取时间最早的交叉（ego 时间索引最小的）
                    first_inter = inter_indices[inter_indices[:, 0].argmin()]
                    intersection_t_ego[b, n] = first_inter[0].float() + 0.5  # 线段中点
                    intersection_t_agent[b, n] = first_inter[1].float() + 0.5
    
    return has_intersection, intersection_t_ego, intersection_t_agent


def detect_intersection_priority(
    ego_traj,       # [B, 1, T, 2] ego 轨迹
    agent_trajs,    # [B, N, T, 2] agent 轨迹
    ego_mask,       # [B, 1, T] ego 有效 mask
    agent_mask,     # [B, N, T] agent 有效 mask
    distance_threshold=3.0,  # 交叉点距离阈值（米）
    multi_step=1,
):
    """
    检测路口冲突的优先级（谁先到冲突点）
    
    核心逻辑：
    1. 检测 ego 轨迹与 agent 轨迹是否在物理空间相交
    2. 如果相交，比较谁先到达交叉点（T 时间索引更小）
    3. T_ego < T_agent: ego 先到 -> +1 (抢行)
       T_ego > T_agent: agent 先到 -> -1 (让行)
       无交叉: 0
    
    Args:
        ego_traj: [B, 1, T, 2] ego 的轨迹
        agent_trajs: [B, N, T, 2] 所有 agent 的轨迹
        ego_mask: [B, 1, T] ego 有效 mask
        agent_mask: [B, N, T] agent 有效 mask
        distance_threshold: 判定为"冲突"的距离阈值
        multi_step: 时间分段数（兼容原接口）
        
    Returns:
        priority: [B, 1, N, multi_step] 优先级
            - +1: ego 先到（抢行）
            - -1: agent 先到（让行）
            - 0: 无冲突
    """
    B, _, T, _ = ego_traj.shape
    N = agent_trajs.shape[1]
    device = ego_traj.device
    
    # Squeeze ego 维度
    ego_traj_sq = ego_traj.squeeze(1)  # [B, T, 2]
    ego_mask_sq = ego_mask.squeeze(1)  # [B, T]
    
    if multi_step > 1:
        seg_len = T // multi_step
        assert seg_len * multi_step == T, f"T={T} 不能被 multi_step={multi_step} 整除"
        
        results = []
        for step in range(multi_step):
            start_t = step * seg_len
            end_t = (step + 1) * seg_len
            
            # 检测这个时间段内的交叉
            has_inter, t_ego, t_agent = find_trajectory_intersections(
                ego_traj_sq[:, start_t:end_t, :],
                agent_trajs[:, :, start_t:end_t, :],
                ego_mask_sq[:, start_t:end_t],
                agent_mask[:, :, start_t:end_t],
            )
            
            # 计算优先级
            # T_ego < T_agent: ego 先到 -> +1
            # T_ego > T_agent: agent 先到 -> -1
            priority = torch.zeros(B, N, device=device)
            priority = torch.where(
                has_inter & (t_ego < t_agent),
                torch.ones_like(priority),   # +1: ego 先到
                priority
            )
            priority = torch.where(
                has_inter & (t_ego >= t_agent),
                -torch.ones_like(priority),  # -1: agent 先到
                priority
            )
            results.append(priority)
        
        # Stack: [B, N, multi_step] -> [B, 1, N, multi_step]
        priority_out = torch.stack(results, dim=-1)[:, None, :, :]
    else:
        # 单时间段
        has_inter, t_ego, t_agent = find_trajectory_intersections(
            ego_traj_sq, agent_trajs, ego_mask_sq, agent_mask
        )
        
        priority = torch.zeros(B, N, device=device)
        priority = torch.where(
            has_inter & (t_ego < t_agent),
            torch.ones_like(priority),
            priority
        )
        priority = torch.where(
            has_inter & (t_ego >= t_agent),
            -torch.ones_like(priority),
            priority
        )
        priority_out = priority[:, None, :, None]  # [B, 1, N, 1]
    
    return priority_out


def generate_intersection_braids(
    ego_traj,           # [B, 1, T, 2] ego 的轨迹
    agent_trajs,        # [B, N, T, 2] 其他 agent 轨迹
    ego_mask,           # [B, 1, T] ego 有效 mask
    agent_mask,         # [B, N, T] agent 有效 mask
    multi_step=1,
):
    """
    生成路口冲突的拓扑标签
    
    这是主入口函数，替代原来的 generate_longitudinal_braids
    
    Args:
        ego_traj: [B, 1, T, 2] ego 的 GT 轨迹 (x, y)
        agent_trajs: [B, N, T, 2] 其他 agent 轨迹 (x, y)
        ego_mask: [B, 1, T] ego 有效 mask
        agent_mask: [B, N, T] agent 有效 mask
        multi_step: 时间分段数
        
    Returns:
        braids: [B, 1, N, multi_step] 拓扑标签
            - +1: ego 先到冲突点（抢行）
            - -1: agent 先到冲突点（让行）
            - 0: 无冲突
    """
    return detect_intersection_priority(
        ego_traj, agent_trajs, ego_mask, agent_mask, 
        multi_step=multi_step
    )


def compute_cumulative_distance(path):
    """
    计算路径的累积弧长
    
    Args:
        path: [B, num_points, 2] 参考路径点
        
    Returns:
        cumulative_s: [B, num_points] 每个点的累积弧长
    """
    # 计算相邻点之间的距离
    diff = path[:, 1:, :] - path[:, :-1, :]  # [B, num_points-1, 2]
    segment_lengths = torch.norm(diff, dim=-1)  # [B, num_points-1]
    
    # 累积求和得到弧长
    zeros = torch.zeros(path.shape[0], 1, device=path.device)
    cumulative_s = torch.cat([zeros, torch.cumsum(segment_lengths, dim=-1)], dim=-1)  # [B, num_points]
    
    return cumulative_s


def project_point_to_path(points, path, path_cumulative_s):
    """
    将点投影到参考路径上，计算 Frenet 坐标 (s, l)
    
    Args:
        points: [B, N, 2] 待投影的点
        path: [B, num_points, 2] 参考路径
        path_cumulative_s: [B, num_points] 路径累积弧长
        
    Returns:
        s: [B, N] 纵向坐标（沿路径的弧长）
        l: [B, N] 横向坐标（到路径的距离，左正右负）
    """
    B, N, _ = points.shape
    num_path_points = path.shape[1]
    
    # 计算每个点到路径上每个点的距离
    # points: [B, N, 1, 2], path: [B, 1, num_points, 2]
    dist = torch.norm(
        points[:, :, None, :] - path[:, None, :, :], 
        dim=-1
    )  # [B, N, num_points]
    
    # 找到最近的路径点索引
    min_dist, nearest_idx = torch.min(dist, dim=-1)  # [B, N]
    
    # 获取对应的 s 坐标
    batch_idx = torch.arange(B, device=path.device)[:, None].expand(B, N)
    s = path_cumulative_s[batch_idx, nearest_idx]  # [B, N]
    
    # 计算 l 坐标（带符号的横向距离）
    nearest_path_points = path[batch_idx, nearest_idx]  # [B, N, 2]
    
    # 计算路径在最近点的切向量（用于判断左右）
    # 使用前后点估计切向量
    next_idx = torch.clamp(nearest_idx + 1, max=num_path_points - 1)
    prev_idx = torch.clamp(nearest_idx - 1, min=0)
    
    next_points = path[batch_idx, next_idx]  # [B, N, 2]
    prev_points = path[batch_idx, prev_idx]  # [B, N, 2]
    
    tangent = next_points - prev_points  # [B, N, 2]
    tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-6)
    
    # 法向量（左手坐标系，左边为正）
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)  # [B, N, 2]
    
    # 从路径点指向目标点的向量
    to_point = points - nearest_path_points  # [B, N, 2]
    
    # l = 投影到法向量上的分量
    l = torch.sum(to_point * normal, dim=-1)  # [B, N]
    
    return s, l


def project_trajectory_to_frenet(trajectory, reference_path):
    """
    将轨迹投影到参考路径的 Frenet 坐标系
    
    Args:
        trajectory: [B, T, 2] 轨迹点 (x, y)
        reference_path: [B, num_points, 2] 参考路径
        
    Returns:
        frenet_traj: [B, T, 2] Frenet 坐标 (s, l)
    """
    # 先计算路径的累积弧长
    path_cumulative_s = compute_cumulative_distance(reference_path)
    
    # 投影
    s, l = project_point_to_path(trajectory, reference_path, path_cumulative_s)
    
    return torch.stack([s, l], dim=-1)


def detect_longitudinal_interaction(
    ego_s, ego_l, ego_mask,
    agent_s, agent_l, agent_mask,
    s_threshold=5.0,
    l_threshold=2.0,
):
    """
    检测纵向交互类型
    
    Args:
        ego_s: [B, T] ego 的 s 坐标序列
        ego_l: [B, T] ego 的 l 坐标序列
        ego_mask: [B, T] ego 有效 mask
        agent_s: [B, N, T] agents 的 s 坐标序列
        agent_l: [B, N, T] agents 的 l 坐标序列
        agent_mask: [B, N, T] agents 有效 mask
        s_threshold: s 方向交互阈值（米）
        l_threshold: l 方向过滤阈值（超过则认为不在同一车道）
        
    Returns:
        interaction_type: [B, N] 交互类型 (+1=超, -1=让, 0=无交互)
    """
    B, N, T = agent_s.shape
    
    # 扩展 ego 坐标以便与 agent 比较
    # ego_s: [B, T] -> [B, 1, T]
    ego_s_exp = ego_s[:, None, :]
    ego_l_exp = ego_l[:, None, :]
    ego_mask_exp = ego_mask[:, None, :]
    
    # 计算每个时刻的 s 差距和 l 差距
    s_diff = ego_s_exp - agent_s  # [B, N, T], 正值表示 ego 在前
    l_diff = torch.abs(ego_l_exp - agent_l)  # [B, N, T]
    
    # 有效性 mask
    valid_mask = ego_mask_exp * agent_mask  # [B, N, T]
    
    # 判断是否在同一车道（l 差距小于阈值）
    same_lane_mask = (l_diff < l_threshold) * valid_mask  # [B, N, T]
    
    # 判断是否存在 s 接近的时刻（可能发生交互）
    s_close_mask = (torch.abs(s_diff) < s_threshold) * same_lane_mask  # [B, N, T]
    
    # 检查是否存在任何可能的交互时刻
    has_interaction = torch.any(s_close_mask, dim=-1)  # [B, N]
    
    # 对于有交互的情况，判断谁先到达交汇区域
    # 策略：在 s 接近的时刻，看 ego 是否在 agent 前面
    
    # 为避免除零，先处理无交互的情况
    s_diff_masked = s_diff * s_close_mask.float()  # 无效位置为 0
    
    # 计算在交互区域内 ego 相对 agent 的平均位置
    num_close_timesteps = s_close_mask.sum(dim=-1).clamp(min=1)  # [B, N]
    avg_s_diff = s_diff_masked.sum(dim=-1) / num_close_timesteps  # [B, N]
    
    # 判断交互类型
    # avg_s_diff > 0: ego 平均在前 -> ego 先通过 -> +1 (超)
    # avg_s_diff < 0: agent 平均在前 -> agent 先通过 -> -1 (让)
    interaction_type = torch.zeros(B, N, device=ego_s.device)
    interaction_type = torch.where(
        has_interaction & (avg_s_diff > 0),
        torch.ones_like(interaction_type),  # +1: 超
        interaction_type
    )
    interaction_type = torch.where(
        has_interaction & (avg_s_diff <= 0),
        -torch.ones_like(interaction_type),  # -1: 让
        interaction_type
    )
    
    return interaction_type


def generate_longitudinal_braids(
    ego_traj,           # [B, 1, T, 2] ego 的轨迹
    agent_trajs,        # [B, N, T, 2] 其他 agent 轨迹
    ego_mask,           # [B, 1, T] ego 有效 mask
    agent_mask,         # [B, N, T] agent 有效 mask
    reference_path,     # [B, num_points, 2] 上游规划的参考路径
    s_threshold=5.0,    # s 方向交互阈值（米）
    l_threshold=2.0,    # l 方向过滤阈值
    multi_step=1,       # 与原 generate_behavior_braids 兼容的多时间段参数
):
    """
    生成纵向交互的拓扑标签（Braid 编码）
    
    相比原版 generate_behavior_braids，这个版本：
    1. 使用 Frenet 坐标系，专注于沿路径的纵向交互
    2. 返回 3 类标签：+1（超）、-1（让）、0（无交互）
    3. 过滤掉不在同一车道的 agent（横向距离大的）
    
    Args:
        ego_traj: [B, 1, T, 2] ego 的 GT 轨迹 (x, y)
        agent_trajs: [B, N, T, 2] 其他 agent 轨迹 (x, y)
        ego_mask: [B, 1, T] ego 有效 mask
        agent_mask: [B, N, T] agent 有效 mask
        reference_path: [B, num_points, 2] 参考路径
        s_threshold: s 方向交互阈值（米），两车 s 差距小于此值认为可能交互
        l_threshold: l 方向过滤阈值（米），两车 l 差距大于此值认为不同车道
        multi_step: 时间分段数（与原函数兼容）
        
    Returns:
        braids: [B, 1, N, multi_step] 拓扑标签
            - +1: ego 先通过（超）
            - -1: agent 先通过（让）
            - 0: 无纵向交互
    """
    B, _, T, _ = ego_traj.shape
    N = agent_trajs.shape[1]
    
    # 计算路径累积弧长（一次计算，多次使用）
    path_cumulative_s = compute_cumulative_distance(reference_path)
    
    # 将 ego 轨迹投影到 Frenet 坐标系
    ego_traj_flat = ego_traj.squeeze(1)  # [B, T, 2]
    ego_s, ego_l = project_point_to_path(
        ego_traj_flat, reference_path, path_cumulative_s
    )  # [B, T]
    
    # 将 agent 轨迹投影到 Frenet 坐标系
    agent_trajs_flat = agent_trajs.reshape(B * N, T, 2)  # [B*N, T, 2]
    reference_path_exp = reference_path[:, None, :, :].expand(B, N, -1, -1).reshape(B * N, -1, 2)
    path_cumulative_s_exp = path_cumulative_s[:, None, :].expand(B, N, -1).reshape(B * N, -1)
    
    agent_s_flat, agent_l_flat = project_point_to_path(
        agent_trajs_flat, reference_path_exp, path_cumulative_s_exp
    )  # [B*N, T]
    agent_s = agent_s_flat.reshape(B, N, T)  # [B, N, T]
    agent_l = agent_l_flat.reshape(B, N, T)  # [B, N, T]
    
    # 处理 mask
    ego_mask_flat = ego_mask.squeeze(1)  # [B, T]
    
    if multi_step > 1:
        # 分时间段处理
        seg_len = T // multi_step
        assert seg_len * multi_step == T, f"T={T} 不能被 multi_step={multi_step} 整除"
        
        results = []
        for step in range(multi_step):
            start_t = step * seg_len
            end_t = (step + 1) * seg_len
            
            interaction = detect_longitudinal_interaction(
                ego_s[:, start_t:end_t], ego_l[:, start_t:end_t], 
                ego_mask_flat[:, start_t:end_t],
                agent_s[:, :, start_t:end_t], agent_l[:, :, start_t:end_t],
                agent_mask[:, :, start_t:end_t],
                s_threshold, l_threshold
            )  # [B, N]
            results.append(interaction)
        
        # Stack: [B, N, multi_step] -> [B, 1, N, multi_step]
        braids = torch.stack(results, dim=-1)[:, None, :, :]
    else:
        # 单时间段
        interaction = detect_longitudinal_interaction(
            ego_s, ego_l, ego_mask_flat,
            agent_s, agent_l, agent_mask,
            s_threshold, l_threshold
        )  # [B, N]
        braids = interaction[:, None, :, None]  # [B, 1, N, 1]
    
    return braids


def generate_longitudinal_braids_simple(
    ego_traj,           # [B, 1, T, 2] ego 的轨迹
    agent_trajs,        # [B, N, T, 2] 其他 agent 轨迹
    ego_mask,           # [B, 1, T] ego 有效 mask  
    agent_mask,         # [B, N, T] agent 有效 mask
    s_threshold=5.0,    # s 方向交互阈值（米）
    l_threshold=2.0,    # l 方向过滤阈值
    multi_step=1,
):
    """
    简化版：当没有 reference_path 时，使用 ego 轨迹本身作为参考路径
    
    这个版本适用于：
    - 暂时没有上游规划输入的情况
    - 快速验证纵向拓扑检测逻辑
    
    Args:
        与 generate_longitudinal_braids 相同，但不需要 reference_path
        
    Returns:
        braids: [B, 1, N, multi_step] 拓扑标签
    """
    B, _, T, _ = ego_traj.shape
    
    # 使用 ego 轨迹作为参考路径
    reference_path = ego_traj.squeeze(1)  # [B, T, 2]
    
    return generate_longitudinal_braids(
        ego_traj, agent_trajs, ego_mask, agent_mask,
        reference_path, s_threshold, l_threshold, multi_step
    )


# ============================================================================
# 测试函数
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Intersection Topology Detection")
    print("=" * 60)
    
    B, N, T = 2, 5, 80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ========== 测试场景 1: 左转让直行 ==========
    # Ego: 从 (0,0) 左转到 (30, 30) 
    # Agent 0: 对向直行，从 (50, 0) 到 (-30, 0)，轨迹会穿过 ego
    # 预期: Agent 先到冲突点 -> -1 (让行)
    
    print("\n[Test 1] 左转让直行")
    ego_traj = torch.zeros(B, 1, T, 2, device=device)
    t_normalized = torch.linspace(0, 1, T, device=device)
    
    # Ego 左转: 曲线轨迹
    ego_traj[:, 0, :, 0] = 30 * t_normalized  # x: 0 -> 30
    ego_traj[:, 0, :, 1] = 30 * t_normalized  # y: 0 -> 30 (斜向走)
    ego_mask = torch.ones(B, 1, T, device=device).bool()
    
    agent_trajs = torch.zeros(B, N, T, 2, device=device)
    agent_mask = torch.zeros(B, N, T, device=device).bool()
    
    # Agent 0: 对向直行，更快速度
    agent_trajs[:, 0, :, 0] = 50 - 80 * t_normalized  # x: 50 -> -30 (快速)
    agent_trajs[:, 0, :, 1] = 5  # y: 固定在 5 (会穿过 ego 的轨迹)
    agent_mask[:, 0, :] = True
    
    # Agent 1: 不相交的远处车辆
    agent_trajs[:, 1, :, 0] = 100 + 10 * t_normalized
    agent_trajs[:, 1, :, 1] = 50
    agent_mask[:, 1, :] = True
    
    braids = generate_intersection_braids(
        ego_traj, agent_trajs, ego_mask, agent_mask
    )
    
    print(f"Output shape: {braids.shape}")  # [B, 1, N, 1]
    print(f"Agent 0 (对向直行): {braids[0, 0, 0, 0].item()}")  # 应该是 -1 (让行) 或 +1
    print(f"Agent 1 (不相交): {braids[0, 0, 1, 0].item()}")    # 应该是 0 (无冲突)
    
    # ========== 测试场景 2: 抢先左转 ==========
    # Ego 更快，先到冲突点
    print("\n[Test 2] 抢先左转")
    ego_traj2 = torch.zeros(B, 1, T, 2, device=device)
    ego_traj2[:, 0, :, 0] = 50 * t_normalized  # x: 更快
    ego_traj2[:, 0, :, 1] = 30 * t_normalized  # y
    
    agent_trajs2 = torch.zeros(B, N, T, 2, device=device)
    agent_mask2 = torch.zeros(B, N, T, device=device).bool()
    
    # Agent 0: 对向直行，但速度慢
    agent_trajs2[:, 0, :, 0] = 80 - 40 * t_normalized  # x: 80 -> 40 (慢)
    agent_trajs2[:, 0, :, 1] = 15  # y
    agent_mask2[:, 0, :] = True
    
    braids2 = generate_intersection_braids(
        ego_traj2, agent_trajs2, ego_mask, agent_mask2
    )
    
    print(f"Agent 0 (慢速对向): {braids2[0, 0, 0, 0].item()}")  # 可能是 +1 (抢行)
    
    # ========== 测试场景 3: 完全平行，无交叉 ==========
    print("\n[Test 3] 平行轨迹，无交叉")
    ego_traj3 = torch.zeros(B, 1, T, 2, device=device)
    ego_traj3[:, 0, :, 0] = 80 * t_normalized  # x
    ego_traj3[:, 0, :, 1] = 0  # y = 0
    
    agent_trajs3 = torch.zeros(B, N, T, 2, device=device)
    agent_mask3 = torch.zeros(B, N, T, device=device).bool()
    
    # Agent 0: 平行行驶
    agent_trajs3[:, 0, :, 0] = 80 * t_normalized + 10  # x
    agent_trajs3[:, 0, :, 1] = 5  # y = 5，平行
    agent_mask3[:, 0, :] = True
    
    braids3 = generate_intersection_braids(
        ego_traj3, agent_trajs3, ego_mask, agent_mask3
    )
    
    print(f"Agent 0 (平行): {braids3[0, 0, 0, 0].item()}")  # 应该是 0 (无冲突)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

