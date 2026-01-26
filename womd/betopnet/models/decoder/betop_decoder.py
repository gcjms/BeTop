'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon Motion Transformer (MTR): 
https://arxiv.org/abs/2209.13508
'''

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from betopnet.models.utils.transformer import transformer_decoder_layer
from betopnet.models.utils.transformer import position_encoding_utils
from betopnet.models.utils.transformer.topo_attention import (
    agent_topo_indexing, map_topo_indexing, apply_topo_attention
)
from betopnet.models.decoder.topo_decoder import TopoFuser, TopoDecoder

from betopnet.models.utils import common_layers
from betopnet.utils import common_utils, loss_utils, motion_utils, topo_utils
from betopnet.config import cfg

import numpy as np


class BeTopDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.end_to_end = self.model_cfg.get('END_TO_END', False)
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        self.num_inter_layers = 2
        self.distinct_anchors = self.model_cfg.get('DISTINCT_ANCHORS', True)
        self.multi_step = 1

        self.num_topo = self.model_cfg.get('NUM_TOPO', 0)

        self.type_dict = {
            'TYPE_VEHICLE':0 , 'TYPE_PEDESTRIAN':1 , 'TYPE_CYCLIST':2
        }

        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        actor_d_model = self.model_cfg.get('ACTOR_D_MODEL', self.d_model)
        # build agent decoders
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=actor_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True
        )
        # build agent decoder MLPs
        self.build_agent_decoder_mlp(actor_d_model)


        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)
        # build map decoders
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True
        )
        # build map decoder MLPs
        self.build_map_decoder_mlp(map_d_model)
        
        # define the dense future prediction layers
        self.build_dense_future_prediction_layers(
            hidden_dim=self.d_model, 
            num_future_frames=self.num_future_frames,
            actor_d_model=actor_d_model
        )

        # build Topo decoders, fusers and cross-layers:
        self.build_topo_layers(
            self.d_model, 
            map_d_model,actor_d_model, 
            0.1, self.num_decoder_layers)

        # define the motion query
        self.intention_points, self.intention_query, \
            self.intention_query_mlps = self.build_motion_query(self.d_model)
        if self.end_to_end:
            self.agent_type_embed = nn.Embedding(3, self.d_model)
            self.agent_init_embed = nn.Embedding(6, self.d_model)

        # define the motion head
        temp_layer = common_layers.build_mlps(
            c_in=self.d_model*2 + map_d_model, 
            mlp_channels=[self.d_model, self.d_model], 
            ret_before_act=True)

        self.query_feature_fusion_layers = nn.ModuleList(
            [copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers
        )

        self.forward_ret_dict = {}
    
    
    def build_agent_decoder_mlp(self, actor_d_model):
        '''
        Building the decoder mlps to align with decoder dim
        for dimension expansion / reduction
        '''
        if actor_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, actor_d_model)
            self.actor_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            temp_r_layer = nn.Linear(actor_d_model, self.d_model)
            self.actor_query_content_mlps_reverse = nn.ModuleList([
                copy.deepcopy(temp_r_layer) for _ in range(self.num_decoder_layers)])
            self.actor_query_embed_mlps = nn.Linear(self.d_model, actor_d_model)

            temp_layer = nn.Linear(self.d_model, actor_d_model)
            self.topo_actor_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
        else:
            self.actor_query_content_mlps_reverse = [None] * self.num_decoder_layers
            self.actor_query_content_mlps = [None] * self.num_decoder_layers
            self.actor_query_embed_mlps = None
        
            self.topo_actor_query_content_mlps = [None] * self.num_decoder_layers
    

    def build_map_decoder_mlp(self, map_d_model):
        '''
        Building the decoder mlps to align with decoder dim
        for dimension expansion / reduction
        '''
        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)

            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.topo_map_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
        else:
            self.map_query_content_mlps = [None] * self.num_decoder_layers
            self.map_query_embed_mlps = None
            self.topo_map_content_mlps = [None] * self.num_decoder_layers

    
    def build_topo_layers(self, d_model, d_map_model, actor_d_model, dropout=0.1, num_decoder_layers=1):

        self.actor_topo_fusers = nn.ModuleList(
            [TopoFuser(actor_d_model, actor_d_model//2, dropout) for _ in range(num_decoder_layers)]
            )
        
        self.map_topo_fusers = nn.ModuleList(
            [TopoFuser(d_map_model, d_map_model//2, dropout) for _ in range(num_decoder_layers)]
            )
        
        self.actor_topo_decoders = nn.ModuleList(
            [TopoDecoder(actor_d_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
            )
        
        self.map_topo_decoders = nn.ModuleList(
            [TopoDecoder(d_map_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
            )


    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames, actor_d_model):
        self.obj_pos_encoding_layer = common_layers.build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.dense_future_head = common_layers.build_mlps(
            c_in=hidden_dim + actor_d_model,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )

        self.future_traj_mlps = common_layers.build_mlps(
            c_in=4 * self.num_future_frames, mlp_channels=[hidden_dim, 
            hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = common_layers.build_mlps(
            c_in=hidden_dim + actor_d_model, mlp_channels=[hidden_dim, 
                hidden_dim, actor_d_model], ret_before_act=True, without_norm=True
        )


    def build_transformer_decoder(self, in_channels, d_model, 
        nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers


    def build_motion_query(self, d_model):

        intention_points = intention_query = intention_query_mlps = None
        intention_points_file = cfg.ROOT_DIR / self.model_cfg.INTENTION_POINTS_FILE
        # for End-to-end decoding, use the 6 K-means anchors instead
        with open(intention_points_file, 'rb') as f:
            intention_points_dict = pickle.load(f)

        intention_points = {}
        for cur_type in self.object_type:
            cur_intention_points = intention_points_dict[cur_type]
            cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
            intention_points[cur_type] = cur_intention_points

        intention_query_mlps = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
        )
        return intention_points, intention_query, intention_query_mlps


    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):
        motion_reg_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True
        )
        motion_cls_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        motion_vel_heads = None 
        return motion_reg_heads, motion_cls_heads, motion_vel_heads


    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        num_center_objects, num_objects, _ = obj_feature.shape

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], 
                self.num_future_frames, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2) 
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, 
        num_objects, self.num_future_frames, 7)
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid
        self.forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

        return ret_obj_feature, ret_pred_dense_future_trajs


    def get_motion_query(self, center_objects_type):
        num_center_objects = len(center_objects_type)

        intention_points = torch.stack([
            self.intention_points[center_objects_type[obj_idx]]
            for obj_idx in range(num_center_objects)], dim=0)
        # (num_query, num_center_objects, 2)
        intention_points = intention_points.permute(1, 0, 2)  

        if self.end_to_end:
            # use embeddings
            agent_type = np.array(
                [self.type_dict[center_objects_type[obj_idx]] 
                for obj_idx in range(num_center_objects)])
    
            agent_type = torch.from_numpy(agent_type).cuda().int()
            embed_type = self.agent_type_embed(agent_type)
            intention_query = self.agent_init_embed.weight
            intention_query = intention_query[:, None, :].repeat(1, num_center_objects, 1) 
            intention_query = intention_query + embed_type[None, :, :] 
        else:
            intention_query = position_encoding_utils.gen_sineembed_for_position(
                intention_points, hidden_dim=self.d_model)

        # (num_query, num_center_objects, D)
        intention_query = self.intention_query_mlps(
            intention_query.view(-1, self.d_model)).view(
                -1, num_center_objects, self.d_model)  
 
        return intention_query, intention_points
    
    def apply_topo_reasoning(
        self, 
        query_feat, kv_feat,
        prev_topo_feat,
        fuse_layer, 
        decoder_layer,
        query_content_pre_mlp=None,
        center_gt_positive_idx=None,
        full_preds=False
        ):
        """
        performing synergistic Topology reasoning
        Args:
            query_feat, kv_feat  [M, B, D], [B, N, D]
            prev_topo_feat, [B, M, N, D]
            fuse_layer, decoder layer: Topo decoders
            center_gt_positive_idx / full_preds:
            Efficient decoding for train-time reasoning 
        """
        
        if query_content_pre_mlp is not None:
            query_feat = query_content_pre_mlp(query_feat)

        query_feat = query_feat.permute(1, 0, 2)
        b = query_feat.shape[0]
        if self.training and not full_preds:
            query_feat = query_feat[torch.arange(b), center_gt_positive_idx][:, None]
     
        src = query_feat
        tgt = kv_feat 
        
        topo_feat = fuse_layer(src, tgt, prev_topo_feat)
        topo_pred = decoder_layer(topo_feat)

        if self.training and not full_preds:
            single_topo_pred = topo_pred
        else:
            single_topo_pred = topo_pred[torch.arange(b), center_gt_positive_idx][:, None]

        return topo_feat, single_topo_pred, topo_pred

    def apply_cross_attention(
        self, query_feat, kv_feat, kv_mask,
        query_pos_feat, kv_pos_feat, 
        pred_query_center, topo_indexing,
        attention_layer,
        query_feat_pre_mlp=None,
        query_embed_mlp=None,
        query_feat_pos_mlp=None,
        is_first=False,
        ): 

        """
        Applying the TopoAttention cross attnetion function
        Args:
            query_feat, query_pos_feat, query_searching_feat  [M, B, D]
            kv_feat, kv_pos_feat  [B, N, D]
            kv_mask [B, N]
            topo_indexing [B, N, N_top]
            attention_layer (func): LocalTransformer Layer (as in EQNet and MTR)
            query_feat_pre_mlp, query_embed_mlp, query_feat_pos_mlp (nn.Linear):
            projections to align decoder dimension
            is_first (bool): whether to concat query pos feature (as in MTR) 
        Returns:
            query_feat: (B, M, D)
        """

        if query_feat_pre_mlp is not None:
            query_feat = query_feat_pre_mlp(query_feat)
        if query_embed_mlp is not None:
            query_pos_feat = query_embed_mlp(query_pos_feat)
        
        d_model = query_feat.shape[-1]
        query_searching_feat = position_encoding_utils.gen_sineembed_for_position(
            pred_query_center, hidden_dim=d_model)
        
        query_feat = apply_topo_attention(
            attention_layer,
            query_feat,
            query_pos_feat,
            query_searching_feat,
            kv_feat,
            kv_pos_feat,
            kv_mask,
            topo_indexing,
            is_first
        )

        if query_feat_pos_mlp is not None:
            query_feat = query_feat_pos_mlp(query_feat)

        return query_feat

    def get_center_gt_idx(
        self, 
        layer_idx,
        pred_scores=None, 
        pred_trajs=None,
        pred_list=None,
        prev_trajs=None,
        prev_dist=None,
        ):
        """
        Calculating GT modality index
        Full: calculating by final displacement of anchors
        E2E: calculating by Winner-Take-All average displacement
        """
        if self.training:
            center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
            center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
            center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
            intention_points = self.forward_ret_dict['intention_points']
            num_center_objects = center_gt_trajs.shape[0]

            if self.end_to_end:
                center_gt_trajs_m = center_gt_trajs_mask.float()[:, None]
                # (num_center_objects, num_query, T)
                dist = (pred_trajs[:, :, :, :2] - center_gt_trajs[:, None, :, :2]).norm(dim=-1)  
                dist = dist * center_gt_trajs_m
                dist = dist.sum(-1) / (center_gt_trajs_m.sum(-1) + (center_gt_trajs_m.sum(-1)==0).float())
                center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
                return center_gt_positive_idx

            center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2] 
            # (num_center_objects, num_query)
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1) 
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)

            if (layer_idx//self.num_inter_layers) * self.num_inter_layers - 1 < 0:
                anchor_trajs = intention_points.unsqueeze(-2)
                select_mask = None
                if pred_list is None:
                    return center_gt_positive_idx, anchor_trajs, dist, select_mask
                if self.distinct_anchors:
                    center_gt_positive_idx, select_mask = motion_utils.select_distinct_anchors(
                        dist, pred_scores, pred_trajs, anchor_trajs
                    )
                return center_gt_positive_idx, anchor_trajs, dist, select_mask

            if self.distinct_anchors:
                # Evolving & Distinct Anchors
                if pred_list is None:
                    # For efficient topo reasoning:
                    unique_layers = set(
                        [(i//self.num_inter_layers)* self.num_inter_layers
                            for i in range(self.num_decoder_layers)]
                    )
                    if layer_idx in unique_layers:
                        anchor_trajs = pred_trajs
                        dist = ((center_gt_trajs[:, None, :, 0:2] - anchor_trajs[..., 0:2]).norm(dim=-1) * \
                             center_gt_trajs_mask[:, None]).sum(dim=-1) 
                    else:
                        anchor_trajs, dist = prev_trajs, prev_dist
                else:
                    anchor_trajs, dist = motion_utils.get_evolving_anchors(
                        layer_idx, self.num_inter_layers, pred_list, 
                        center_gt_goals, intention_points, 
                        center_gt_trajs, center_gt_trajs_mask, 
                        )

                center_gt_positive_idx, select_mask = motion_utils.select_distinct_anchors(
                    dist, pred_scores, pred_trajs, anchor_trajs
                )
        else:
            center_gt_positive_idx = None
            anchor_trajs, dist = None, None
            select_mask=None

        return center_gt_positive_idx, anchor_trajs, dist, select_mask

    def apply_transformer_decoder(
        self, center_objects_feature, center_objects_type,
        obj_feature, obj_mask, obj_pos, 
        map_feature, map_mask, map_pos):
        """
        ============================================================================
        BeTop Decoder 核心函数 - 详细中文注释版
        ============================================================================
        
        【输入参数说明】
        - center_objects_feature: [B, D] 中心对象（待预测目标）的特征，来自 Encoder
        - center_objects_type: [B] 中心对象的类型（车辆/行人/骑行者）
        - obj_feature: [B, N_agents, D] 场景中所有障碍物的特征
        - obj_mask: [B, N_agents] 障碍物有效性掩码
        - obj_pos: [B, N_agents, 3] 障碍物最后时刻位置 (x, y, z)
        - map_feature: [B, N_map, D] 地图车道线特征
        - map_mask: [B, N_map] 地图有效性掩码
        - map_pos: [B, N_map, 3] 地图车道线中心位置
        
        【输出】
        - pred_list: 每层 Decoder 的预测结果列表
          - pred_scores: [B, 64] 每条轨迹的置信度分数
          - pred_trajs: [B, 64, 80, 7] 64条候选轨迹，每条80帧，7维状态
        ============================================================================
        """
        
        # ========== Step 1: 初始化 64 个 Motion Query ==========
        # intention_query: [64, B, D] - 64个意图点的位置编码，作为 Decoder 的 Query
        # intention_points: [64, B, 2] - 64个聚类中心点的 xy 坐标
        intention_query, intention_points = self.get_motion_query(center_objects_type)
        
        # query_content: [64, B, D] - Query 的内容特征，初始化为零
        # 这个会在每层 Decoder 中逐步更新，累积上下文信息
        query_content = torch.zeros_like(intention_query)
        
        # 保存 intention_points 用于后续损失计算
        # 转换为 [B, 64, 2] 的格式
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  

        dim = query_content.shape[-1]        # D = 512
        num_center_objects = query_content.shape[1]  # B = batch中待预测对象数量
        num_query = query_content.shape[0]   # 64 = Motion Query 数量

        # 将中心对象特征复制 64 份，每个 Query 都能访问
        # [1, B, D] -> [64, B, D]
        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  

        # ========== Step 2: 初始化各种状态变量 ==========
        base_map_idxs = None       # 基础地图索引（自车周围的固定区域）
        map_topo_feat = None       # 地图拓扑特征（跨层传递）
        actor_topo_feat = None     # 障碍物拓扑特征（跨层传递）
        center_gt_positive_idx = None  # GT对应的模态索引（训练时使用）
        pred_scores, pred_trajs = None, None  # 上一层的预测结果
        anchor_trajs, anchor_dist = None, None  # 用于 Distinct Anchor 选择

        # 预测轨迹的中间路径点，初始化为 intention_points
        # [B, 64, 1, 2] - 后续会更新为预测轨迹的每一帧
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        
        # 动态 Query 中心，初始化为意图点位置
        # [64, B, 2] - 后续会更新为预测轨迹的终点
        dynamic_query_center = intention_points

        # ========== Step 3: 生成位置编码 ==========
        # 地图位置编码: [N_map, B, D]
        map_pos_p = map_pos.permute(1, 0, 2)[:, :, 0:2]
        map_pos_embed = position_encoding_utils.gen_sineembed_for_position(map_pos_p, 
            hidden_dim=map_feature.shape[-1])

        # 障碍物位置编码: [N_agents, B, D]
        obj_pos_p = obj_pos.permute(1, 0, 2)[:, :, 0:2]
        obj_pos_embed = position_encoding_utils.gen_sineembed_for_position(obj_pos_p,
             hidden_dim=obj_feature.shape[-1])

        pred_list = []  # 存储每层的预测结果

        # ============================================================================
        # ================ Step 4: Decoder 主循环 (共6层) ================
        # ============================================================================
        for layer_idx in range(self.num_decoder_layers):
            
            # ---------- Step 4.1: 训练时获取 GT 对应的模态索引 ----------
            # 用于计算损失时确定哪个模态最接近真值
            if not self.end_to_end:
                center_gt_positive_idx, anchor_trajs, anchor_dist, _ = self.get_center_gt_idx(
                    layer_idx, pred_scores, pred_trajs, prev_trajs=anchor_trajs, prev_dist=anchor_dist
                )

            # ============================================================================
            # ---------- Step 4.2: Agent 拓扑推理 (BeTop 核心创新) ----------
            # ============================================================================
            # 【目的】预测每个 Query 和每个障碍物的拓扑关系分数
            # 【输入】query_content: [64, B, D], obj_feature: [B, N_agents, D]
            # 【输出】
            #   - actor_topo_feat: [B, 64, N_agents, D] - 拓扑交互特征（传递到下一层）
            #   - actor_topo_preds: [B, 1, N_agents, 1] - 当前 GT 模态的拓扑分数（计算损失用）
            #   - full_actor_topo_preds: [B, 64, N_agents, 1] - 所有模态的拓扑分数（选择 TopK 用）
            actor_topo_feat, actor_topo_preds, full_actor_topo_preds  = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=obj_feature,
                prev_topo_feat=actor_topo_feat,  # 上一层的拓扑特征，实现跨层信息传递
                fuse_layer=self.actor_topo_fusers[layer_idx], 
                decoder_layer=self.actor_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_actor_query_content_mlps[layer_idx],
                center_gt_positive_idx=center_gt_positive_idx,
                full_preds=True  # 需要完整的 64×N 预测用于索引
            )
            
            # ---------- Step 4.3: Agent 拓扑索引选择 ----------
            # 根据拓扑分数，为每个 Query 选择 Top-K 个最相关的障碍物
            # 【输入】full_actor_topo_preds: [B, 64, N_agents, 1] - 拓扑分数
            # 【输出】pred_agent_topo_idx: [B, 64, 32] - 选中的障碍物索引（32个）
            pred_agent_topo_idx = agent_topo_indexing(
                full_actor_topo_preds, obj_mask, max_agents=self.num_topo)  # num_topo=32
            
            # ============================================================================
            # ---------- Step 4.4: Map 拓扑推理 ----------
            # ============================================================================
            # 与 Agent 拓扑推理类似，但针对地图车道线
            map_topo_feat, map_topo_preds, full_map_topo_preds = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=map_feature,
                prev_topo_feat=map_topo_feat,
                fuse_layer=self.map_topo_fusers[layer_idx], 
                decoder_layer=self.map_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_map_content_mlps[layer_idx],
                center_gt_positive_idx=center_gt_positive_idx,
                full_preds=False  # 地图拓扑可以简化，不需要完整预测
            )
            
            # ---------- Step 4.5: Map 拓扑索引选择 ----------
            # 结合两种策略选择地图车道线：
            # 1. 基础区域：自车周围固定范围的车道线 (256条)
            # 2. 动态路径：预测轨迹沿途的车道线 (128条)
            pred_map_topo_idxs, base_map_idxs = map_topo_indexing(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,  # 用预测轨迹来选择沿途车道线
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,  # [30, 0] 前方30米
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,  # 128
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,  # 256
                base_map_idxs=base_map_idxs,
                num_query=num_query
            )

            # ============================================================================
            # ---------- Step 4.6: Topo Cross Attention (Agent) ----------
            # ============================================================================
            # 【关键】这里使用 pred_agent_topo_idx 来稀疏化 Attention
            # Query 只和拓扑分数最高的 32 个障碍物进行交互，而不是全部 N 个
            # 【输入】
            #   - query_feat: [64, B, D] 当前 Query 特征
            #   - kv_feat: [B, N_agents, D] 障碍物特征
            #   - topo_indexing: [B, 64, 32] 选中的障碍物索引
            # 【输出】agent_query_feature: [64, B, D] 更新后的 Query 特征
            agent_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=obj_feature, kv_mask=obj_mask,
                query_pos_feat=intention_query, kv_pos_feat=obj_pos_embed, 
                pred_query_center=dynamic_query_center,  # 用当前预测终点作为 Query 的搜索中心
                topo_indexing=pred_agent_topo_idx,  # 【关键】拓扑索引，实现稀疏 Attention
                attention_layer=self.obj_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.actor_query_content_mlps[layer_idx],
                query_embed_mlp=self.actor_query_embed_mlps,
                query_feat_pos_mlp=self.actor_query_content_mlps_reverse[layer_idx],
                is_first=layer_idx==0,  # 第一层需要特殊处理位置编码
            ) 

            # ---------- Step 4.7: Topo Cross Attention (Map) ----------
            # 与 Agent 处理类似，但针对地图车道线
            map_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=map_feature, kv_mask=map_mask,
                query_pos_feat=intention_query, kv_pos_feat=map_pos_embed, 
                pred_query_center=dynamic_query_center, topo_indexing=pred_map_topo_idxs,
                attention_layer=self.map_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_mlp=self.map_query_embed_mlps,
                is_first=layer_idx==0,   
            ) 

            # ============================================================================
            # ---------- Step 4.8: 特征融合 ----------
            # ============================================================================
            # 拼接三种特征：中心对象 + Agent交互 + Map交互
            # [64, B, D] + [64, B, D] + [64, B, D_map] -> [64, B, D*2+D_map]
            query_feature = torch.cat([
                center_objects_feature, agent_query_feature, map_query_feature
                ], dim=-1)
         
            # 通过 MLP 融合并降维回 D
            # [64*B, D*2+D_map] -> [64*B, D] -> [64, B, D]
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            # ============================================================================
            # ---------- Step 4.9: 预测头 (Motion Head) ----------
            # ============================================================================
            # 将 Query 特征转换为具体的轨迹预测
            # query_content_t: [B*64, D] - 展平用于批量 MLP
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            
            # 预测置信度分数: [B*64, 1] -> [B, 64]
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            
            # 预测轨迹: [B*64, 80*7] -> [B, 64, 80, 7]
            # 7维 = (x, y, σx, σy, ρ, vx, vy) - GMM参数 + 速度
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            # 保存当前层的预测结果
            pred_list.append([pred_scores, pred_trajs, actor_topo_preds, map_topo_preds])
            
            # ============================================================================
            # ---------- Step 4.10: 更新状态用于下一层 ----------
            # ============================================================================
            # 用当前层的预测结果更新 Query 的搜索中心
            # 这实现了"迭代细化"：下一层的 Attention 会基于更准确的预测位置
            
            # pred_waypoints: [B, 64, 80, 2] - 预测轨迹的所有点（用于选择沿途地图）
            pred_waypoints = pred_trajs.detach().clone()[:, :, :, 0:2]
            
            # dynamic_query_center: [64, B, 2] - 预测轨迹的终点（用于下一层 Attention）
            dynamic_query_center = pred_trajs.detach().clone()[:, :, -1, 0:2].contiguous().permute(1, 0, 2)  

        assert len(pred_list) == self.num_decoder_layers  # 确保 6 层都有输出
        return pred_list
    
    def build_topo_gt(self, gt_trajs, gt_valid_mask, multi_step=1):
        gt_trajs = gt_trajs * gt_valid_mask[..., None].float()
        map_pos = self.forward_ret_dict['map_polylines']
        map_mask = self.forward_ret_dict['map_mask']
        polyline_mask = self.forward_ret_dict['map_polylines_mask']

        tgt_trajs = self.forward_ret_dict['obj_trajs_future_state'].cuda()
        tgt_trajs_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()
        tgt_trajs = tgt_trajs * tgt_trajs_mask[..., None].float()

        actor_topo = topo_utils.generate_behavior_braids(gt_trajs[:, None, :, :2], tgt_trajs[..., :2], 
                gt_valid_mask[:, None], tgt_trajs_mask, multi_step)
        actor_topo_mask = torch.any(tgt_trajs_mask, dim=-1)[:, None, :]

        map_topo = topo_utils.generate_map_briads(gt_trajs[:, None, :, :2], map_pos[:, :, :, :2], 
                gt_valid_mask[:, None, :], polyline_mask, multi_step)
        map_topo_mask = map_mask[:, None, :]

        return actor_topo, actor_topo_mask, map_topo, map_topo_mask
    
    def topo_loss(
        self, 
        actor_topo, actor_topo_mask, map_topo, map_topo_mask,
        actor_topo_pred, map_topo_pred,
        ):
        """
        ============================================================================
        拓扑损失函数 (BeTop 核心创新)
        ============================================================================
        
        【作用】
        监督模型学习预测正确的拓扑关系（如"自车会从左边超过障碍物A"）
        
        【输入】
        - actor_topo: [B, 1, N_agents, T] Ground Truth 拓扑标签（Braid 编码）
          - 编码了自车与其他障碍物的交互模式（如轨迹交叉顺序）
        - actor_topo_mask: [B, 1, N_agents] 有效障碍物掩码
        - actor_topo_pred: [B, 1, N_agents, 1] 模型预测的拓扑分数
        - map_topo / map_topo_pred: 地图车道线的拓扑关系（类似）
        
        【实现细节】
        - top_k=True, top_k_ratio=0.25: 只计算 Top 25% 难例的 Loss（Hard Example Mining）
        - 这样做可以让模型更关注预测错误的样本，提高学习效率
        
        【返回】
        - actor_topo_loss: 障碍物拓扑损失
        - map_topo_loss: 地图拓扑损失
        ============================================================================
        """
        actor_topo_loss = loss_utils.topo_loss(actor_topo_pred, actor_topo.detach(), 
            actor_topo_mask.float().detach(), top_k=True, top_k_ratio=0.25)
        map_topo_loss = loss_utils.topo_loss(map_topo_pred, map_topo[..., None].detach(), 
            map_topo_mask.float().detach(), top_k=True, top_k_ratio=0.25)

        return  actor_topo_loss, map_topo_loss

    def get_decoder_loss(self, tb_pre_tag=''):
        """
        ============================================================================
        Decoder 主损失函数 - 计算轨迹预测的所有损失
        ============================================================================
        
        【作用】
        计算 Decoder 每一层的损失，包括：
        1. loss_reg_gmm: GMM 回归损失（预测轨迹与真值的距离）
        2. loss_reg_vel: 速度回归损失（预测速度与真值的L1距离）
        3. loss_cls: 分类损失（预测哪个模态最接近真值）
        4. loss_topo: 拓扑损失（BeTop 创新，预测交互关系）
        
        【损失权重】(来自配置文件)
        - weight_reg = 1.0
        - weight_vel = 0.2 (速度权重较小，因为速度比位置容易预测)
        - weight_cls = 1.0
        - weight_top = 100 (拓扑权重很大，说明很重要！)
        
        【返回】
        - total_loss: 所有层损失的平均值
        - tb_dict: TensorBoard 日志字典
        - disp_dict: 显示字典（用于打印）
        ============================================================================
        """
        # 获取 Ground Truth 数据
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()  # [B, 80, 4] (x, y, vx, vy)
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()  # [B, 80] 有效帧掩码
        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()  # [B] 最后有效帧索引
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']  # 每层 Decoder 的预测结果
        intention_points = self.forward_ret_dict['intention_points']  # [B, 64, 2] 意图点

        num_center_objects = center_gt_trajs.shape[0]
        # 提取真实终点位置，用于确定哪个模态最接近真值
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # [B, 2]
        
        # 构建拓扑 Ground Truth（用于 topo_loss）
        actor_topo, actor_topo_mask, map_topo, map_topo_mask = self.build_topo_gt(
            center_gt_trajs, center_gt_trajs_mask, self.multi_step)
    
        tb_dict = {}
        disp_dict = {}
        total_loss = 0
        
        # ========== 遍历每层 Decoder 计算损失 ==========
        for layer_idx in range(self.num_decoder_layers):
     
            pred_scores, pred_trajs, actor_topo_preds, map_topo_preds = pred_list[layer_idx]  
            
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            center_gt_positive_idx,_,_,select_mask = self.get_center_gt_idx(
                    layer_idx, pred_scores, pred_trajs, pred_list=pred_list
                )

            loss_a_topo, loss_m_topo = self.topo_loss(
                actor_topo, actor_topo_mask, map_topo, map_topo_mask,
                actor_topo_preds, map_topo_preds,
            )
            loss_topo =  loss_a_topo + loss_m_topo

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            bce_target = torch.zeros_like(pred_scores)
            bce_target[torch.arange(num_center_objects), center_gt_positive_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(input=pred_scores, target=bce_target, reduction='none')
            loss_cls = (loss_cls * select_mask).sum(dim=-1)

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)
            weight_top = self.model_cfg.LOSS_WEIGHTS.get('top', 100)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel +\
                loss_cls.sum(dim=-1) * weight_cls + weight_top * loss_topo
           
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_top'] = loss_topo.mean().item() * weight_top
   
            if layer_idx + 1 == self.num_decoder_layers:
                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=tb_pre_tag
                )
                tb_dict.update(layer_tb_dict_ade)
                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / self.num_decoder_layers

        return total_loss, tb_dict, disp_dict

    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):
        """
        ============================================================================
        Dense Future Prediction Loss - 场景中所有障碍物的未来轨迹预测损失
        ============================================================================
        
        【作用】
        这个函数计算的是"场景中所有障碍物"的未来轨迹预测损失，不只是中心对象！
        
        【为什么需要这个损失？】
        1. 预测其他障碍物的未来轨迹可以帮助模型理解"社会交互"
        2. 这些预测的特征会被融合到中心对象的预测中（见 apply_dense_future_prediction）
        3. 类似于"我预测你会往左走，所以我应该往右走"
        
        【输入（从 forward_ret_dict 获取）】
        - obj_trajs_future_state: [B, N_agents, 80, 4] 所有障碍物的真实未来轨迹
        - obj_trajs_future_mask: [B, N_agents, 80] 有效帧掩码
        - pred_dense_trajs: [B, N_agents, 80, 7] 模型预测的未来轨迹
          - 7维 = (x, y, σx, σy, ρ, vx, vy) GMM 参数 + 速度
        
        【计算内容】
        - loss_reg_gmm: GMM 位置回归损失（高斯混合模型负对数似然）
        - loss_reg_vel: 速度回归损失（L1 Loss）
        
        【返回】
        - loss_reg: 平均损失（对所有有效障碍物取平均）
        ============================================================================
        """
        # 获取数据
        obj_trajs_future_state = self.forward_ret_dict['obj_trajs_future_state'].cuda()  # [B, N, 80, 4]
        obj_trajs_future_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()  # [B, N, 80]
        pred_dense_trajs = self.forward_ret_dict['pred_dense_trajs']  # [B, N, 80, 7]
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        # 拆分预测结果：GMM 参数 (x, y, σx, σy, ρ) 和速度 (vx, vy)
        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        # 速度回归损失: L1 Loss
        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        # 位置回归损失: GMM 负对数似然
        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        # fake_scores: 因为这里不是多模态预测，所以分数全为 0
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)

        # 展平维度以适配 loss 函数
        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        
        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        # 合并损失并对有效障碍物取平均
        loss_reg = loss_reg_vel + loss_reg_gmm
        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0  # 有任意有效帧的障碍物
        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
        loss_reg = loss_reg.mean()

        if tb_dict is None:
            tb_dict = {}
        if disp_dict is None:
            disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        return loss_reg, tb_dict, disp_dict

    def get_loss(self, tb_pre_tag=''):
        """
        ============================================================================
        总损失函数入口 - 训练时调用
        ============================================================================
        
        【作用】
        汇总所有损失，返回用于反向传播的总损失值
        
        【损失组成】
        total_loss = loss_decoder + loss_dense_prediction
        
        其中:
        - loss_decoder: Decoder 损失 (GMM + Vel + Cls + Topo)
        - loss_dense_prediction: 场景中所有障碍物的未来轨迹预测损失
        
        【返回】
        - total_loss: 用于 backward() 的总损失
        - tb_dict: TensorBoard 日志（用于可视化训练曲线）
        - disp_dict: 显示字典（用于命令行打印）
        ============================================================================
        """
        # Decoder 损失：轨迹预测 + 分类 + 拓扑
        loss_decoder, tb_dict, disp_dict = self.get_decoder_loss(tb_pre_tag=tb_pre_tag)
        
        # Dense Future 损失：场景中所有障碍物的预测
        loss_dense_prediction, tb_dict, disp_dict = self.get_dense_future_prediction_loss(
            tb_pre_tag=tb_pre_tag, tb_dict=tb_dict, disp_dict=disp_dict)

        # 汇总
        total_loss = loss_decoder + loss_dense_prediction
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()
        disp_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, pred_list):
        """
        ============================================================================
        推理后处理 - 从 64 条候选轨迹中选出最终的 6 条
        ============================================================================
        
        【作用】
        推理时调用，通过 NMS（非极大值抑制）筛选出多样化的最终预测轨迹
        
        【流程】
        1. 取最后一层 Decoder 的预测结果（最精细的预测）
        2. 将分数通过 sigmoid 归一化到 [0, 1]
        3. 如果候选数 > 输出数（64 > 6），则用 NMS 筛选
        
        【NMS 策略】
        - 按分数排序，贪心选择
        - 如果新轨迹的终点与已选轨迹终点距离 < 阈值，则跳过
        - 目的：保证输出轨迹的多样性（不要 6 条轨迹都差不多）
        
        【返回】
        - pred_scores_final: [B, 6] 最终 6 条轨迹的分数
        - pred_trajs_final: [B, 6, 80, 7] 最终 6 条轨迹
        - selected_idxs: [B, 6] 被选中的轨迹索引（用于调试）
        ============================================================================
        """
        # 取最后一层的预测结果
        pred_scores, pred_trajs, _, _ = pred_list[-1]
        pred_scores = torch.sigmoid(pred_scores)  # [B, 64] -> 归一化到 [0, 1]

        # 如果候选数 > 最终输出数，需要 NMS 筛选
        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes
            # NMS: 64 条 -> 6 条
            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.inference_distance_nms(
                pred_scores, pred_trajs
            )
        else:
            # 无需筛选，直接返回
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores
            selected_idxs = None
   
        return pred_scores_final, pred_trajs_final, selected_idxs

    def forward(self, batch_dict):
        """
        ============================================================================
        BeTopDecoder 前向传播入口
        ============================================================================
        
        【作用】
        接收 Encoder 的输出，执行完整的 Decoder 流程，输出轨迹预测
        
        【输入 batch_dict（来自 Encoder）】
        - obj_feature: [B, N_agents, D] 障碍物特征
        - map_feature: [B, N_map, D] 地图特征
        - center_objects_feature: [B, D] 中心对象特征
        - obj_pos / map_pos: 位置信息
        - input_dict: 原始输入数据（包含 GT）
        
        【处理流程】
        1. 特征投影：将 Encoder 特征投影到 Decoder 维度
        2. Dense Future Prediction：预测场景中所有障碍物的未来轨迹
        3. Transformer Decoder：6 层 Decoder 迭代预测
        4. 后处理（推理时）：NMS 选出最终 6 条轨迹
        
        【输出（写入 batch_dict）】
        训练时：pred_list 存入 forward_ret_dict（用于计算 Loss）
        推理时：pred_scores, pred_trajs 存入 batch_dict（用于评估）
        ============================================================================
        """
        input_dict = batch_dict['input_dict']
        
        # 从 Encoder 输出中提取特征
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # ========== Step 1: 特征投影 ==========
        # 将 Encoder 输出投影到 Decoder 的维度（可能不同）
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)  # [B, D] -> [B, D_decoder]
        
        # 只对有效障碍物做投影（节省计算）
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        # 只对有效地图做投影
        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # ========== Step 2: Dense Future Prediction ==========
        # 预测场景中所有障碍物的未来轨迹，并将预测特征融合回 obj_feature
        # 这一步让模型能"预判"其他障碍物的行为
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )
        
        # ========== Step 3: Transformer Decoder ==========
        if self.training:
            self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
            self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
            self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
        
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos
        )

        self.forward_ret_dict['pred_list'] = pred_list

        if not self.training:
            pred_scores, pred_trajs, selected_idxs = self.generate_final_prediction(pred_list=pred_list)
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs
            batch_dict['selected_idxs'] = selected_idxs

        else:
            self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
            self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']

            self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

            self.forward_ret_dict['map_pos'] = map_pos
            self.forward_ret_dict['map_mask'] = map_mask

            self.forward_ret_dict['map_polylines'] = batch_dict['map_polylines']
            self.forward_ret_dict['map_polylines_mask'] = batch_dict['map_polylines_mask']
            self.forward_ret_dict['map_mask'] = batch_dict['map_mask']

        return batch_dict
