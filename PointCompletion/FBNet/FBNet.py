'''
modified:
Release Version for FBNet: Feedback network for point cloud completion
'''

import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "..","utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_utils import furthest_point_sample, grouping_operation, ball_query, three_interpolate
from pointnet2_utils import gather_operation

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    _, idx = sqrdists.topk(nsample, largest=False)
    return idx.int()

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        self.af = nn.LeakyReLU(negative_slope=0.2)
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(self.af)
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def group_local(xyz, k=20, return_idx=False):
    """
    Input:
        x: point cloud, [B, 3, N]
    Return:
        group_xyz: [B, 3, N, K]
    """
    xyz = xyz.transpose(2, 1).contiguous()
    idx = query_knn_point(k, xyz, xyz)
    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz.permute(0, 3, 1, 2)
    if return_idx:
        return group_xyz, idx

    return group_xyz


class point_shuffler(nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C1, scale*N]
    """   
    def __init__(self, scale=2):
        super(point_shuffler, self).__init__()
        self.scale = scale
    def forward(self, inputs: "(B, channel_num, N)"):
        if self.scale == 1:
            ou = inputs
        else:
            B, C, N = inputs.shape
            x = inputs.permute([0,2,1])
            ou = x.reshape([B, N, self.scale, C//self.scale])
            ou = ou.reshape([B, N*self.scale, C//self.scale]).permute([0,2,1])
        

        return ou


class NodeShuffle(nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C1, scale*N]
    """
    def __init__(self, input_channel, output_channel, neighbor_num=16, scale=2):
        super(NodeShuffle, self).__init__()
        self.num_neigh = neighbor_num
        self.scale = scale
        self.ou_c = output_channel

        self.edge_scale = min(scale,4)
        self.edge_conv = EdgeConv(input_channel, input_channel*self.edge_scale, neighbor_num)
        self.mlp = MLP_CONV(in_channel=input_channel*self.edge_scale, layer_dims=[output_channel*scale], bn=True)
        self.pt_shuffle = point_shuffler(scale = scale)


    def forward(self, inputs: "(B, channel_num, N)"):
        """
        Return:
            outputs: (B, channel_num, rN)
        """
        batch_size, dims, num_points = inputs.shape

        outputs = self.edge_conv(inputs)
        outputs = self.mlp(outputs)
        outputs = self.pt_shuffle(outputs)


        return outputs

class EdgeConv(torch.nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C2, N]
    """

    def __init__(self, input_channel, output_channel, k):
        super(EdgeConv, self).__init__()
        self.num_neigh = k

        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel, kernel_size=1)
        )

    def forward(self, inputs):
        batch_size, dims, num_points = inputs.shape
        if self.num_neigh is not None:
            neigh_feature = group_local(inputs, k=self.num_neigh).contiguous()
            central_feat = inputs.unsqueeze(dim=3).repeat(1, 1, 1, self.num_neigh)
        else:
            central_feat = torch.zeros(batch_size, dims, num_points, 1).to(inputs.device)
            neigh_feature = inputs.unsqueeze(-1)
        edge_feature = central_feat - neigh_feature
        feature = torch.cat((edge_feature, central_feat), dim=1)
        feature = self.conv(feature)
        central_feature = feature.max(dim=-1, keepdim=False)[0]
        return central_feature


class AdaptGraphPooling(nn.Module):
    def __init__(self, pooling_rate, in_channel, neighbor_num, dim=64):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(dim, 3 + in_channel, 1)
        )

    def forward(self, vertices: "(bs, 3, vertice_num)",
                feature_map: "(bs, channel_num, vertice_num)",
                idx=False):
        """
        
        Return:
            vertices_pool: (bs, 3, pool_vertice_num),
            feature_map_pool: (bs, channel_num, pool_vertice_num)
        """

        bs, _, vertice_num = vertices.size()
        new_npoints = int(vertice_num*1.0 / self.pooling_rate+0.5)
        key_points_idx = furthest_point_sample(vertices.transpose(2,1).contiguous(), new_npoints)
        key_point = gather_operation(vertices, key_points_idx)
        key_feat = gather_operation(feature_map, key_points_idx)

        key_point_idx = query_knn(self.neighbor_num, vertices.transpose(2,1).contiguous(), key_point.transpose(2,1).contiguous(), include_self=True)

        group_point = grouping_operation(vertices, key_point_idx)
        group_feat = grouping_operation(feature_map, key_point_idx)

        qk_rel = key_feat.reshape((bs, -1, new_npoints, 1)) - group_feat
        pos_rel = key_point.reshape((bs, -1, new_npoints, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding) # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1) # b, in_channel + 3, n, n_knn
        new_xyz_weight = sample_weight[:,:3,:,:]  # b, 3, n, n_knn
        new_feture_weight = sample_weight[:,3:,:,:]  # b, in_channel, n, n_knn

        group_feat = group_feat + pos_embedding  #
        new_feat = einsum('b c i j, b c i j -> b c i', new_feture_weight, group_feat)
        new_point = einsum('b c i j, b c i j -> b c i', new_xyz_weight, group_point)

        return new_point, new_feat


# Hierarchical Graph-based Network
class HGNet(nn.Module):
    def __init__(self, num_pc=128, g_feat_dim=1024,using_max=True):
        super(HGNet, self).__init__()

        self.using_max = using_max
        self.num_pc = num_pc
        pool_num = 2048

        self.out_channel = g_feat_dim//2

        # HGNet econder
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.graph_pooling_1 = AdaptGraphPooling(4, 64, 16)
        self.gcn_2 = EdgeConv(64, 128, 16)
        self.graph_pooling_2 = AdaptGraphPooling(2, 128, 16)
        self.gcn_3 = EdgeConv(128, 512, 16)

        # Fully-connected decoder
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 3*num_pc)
        )


    def forward(self, inputs):
        device = inputs.device
        batch_size = inputs.size(0)
        x1 = self.gcn_1(inputs)
        vertices_pool_1, x1 = self.graph_pooling_1(inputs, x1)

        # B x 128 x 512
        x2 = self.gcn_2(x1)
        vertices_pool_2, x2 = self.graph_pooling_2(vertices_pool_1, x2)

        # B x 256 x 256
        x3 = self.gcn_3(x2)

        # Global feature generating B*1024
        feat_max = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1)
        feat_avg = F.adaptive_avg_pool1d(x3, 1).view(batch_size, -1)
        feat_gf = torch.cat((feat_max, feat_avg), dim=1)
        
        # Decoder coarse input
        coarse_pcd = self.fc(feat_gf).reshape(batch_size, -1, self.num_pc)

        return coarse_pcd, feat_max


class CrossTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(CrossTransformer, self).__init__()
        self.n_knn = n_knn

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, in_channel, 1)
        )

    def forward(self, pcd, feat, pcd_feadb, feat_feadb):
        """
        Args:
            pcd: (B, 3, N)
            feat: (B, in_channel, N)
            pcd_feadb: (B, 3, N2)
            feat_feadb: (B, in_channel, N2)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        b, _, num_point = pcd.shape


        fusion_pcd = torch.cat((pcd, pcd_feadb), dim=2)
        fusion_feat = torch.cat((feat, feat_feadb), dim=2)

        key_point = pcd
        key_feat = feat

        # Preception processing between pcd and fusion_pcd
        key_point_idx = query_knn(self.n_knn, fusion_pcd.transpose(2,1).contiguous(), key_point.transpose(2,1).contiguous(), include_self=True)

        group_point = grouping_operation(fusion_pcd, key_point_idx)
        group_feat = grouping_operation(fusion_feat, key_point_idx)

        
        qk_rel = key_feat.reshape((b, -1, num_point, 1)) - group_feat
        pos_rel = key_point.reshape((b, -1, num_point, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding) # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1) # b, in_channel + 3, n, n_knn

        group_feat = group_feat + pos_embedding  #
        refined_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)
        
        return refined_feat


class FBAC_BLOCK(nn.Module):
    def __init__(self, up_factor=2, cycle_num=1):
        """
        des: Feedback-Aware Completion block
        input: point cloud: B, 3, N
        param: up_factor: up-sampling ratio
               cycle_num: number of time steps
        return: point cloud: B, 3, N * up_factor
        
        """
        super(FBAC_BLOCK, self).__init__()
        # self.cyc_num = cyc_num
        self.up_factor = up_factor
        # self.gf_mode = gf_mode
        # self.weight = weight


        self.nodeshuffle = NodeShuffle(128, 128, neighbor_num=8, scale=up_factor)
        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[128, 64, 3])

        self.ext = EdgeConv(3, 128, 16)
        
        self.mlp = MLP_CONV(in_channel=128 * 2, layer_dims=[256, 128])


        self.fb_exploit = CrossTransformer(in_channel=128, dim=64)


        self.up_sampler = nn.Upsample(scale_factor=up_factor)

        # self.alphas = nn.Embedding(cycle_num,1,_weight=torch.ones(cycle_num,1))
        # self.sigmoid = nn.Sigmoid()


    def forward(self, pcd, pcd_next, feat_next, cycle=0):
        """
        Args:
            pcd: Tensor, (B, 3, N_prev)
            pcd_next: Tensor, (B, 3, N_next) 
            K_next: Tensor, (B, 128, N_next)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
        """
        
        b, C, n_prev = pcd.shape
        
        # Step 1: Feature Extraction
        feat = self.ext(pcd)
        feat = self.mlp(torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1))



        # Step 2: Feedback Exploitation
        if pcd_next is None:
            pcd_next, feat_next = pcd, feat 
        feat = self.fb_exploit(pcd, feat, pcd_next, feat_next)

        # Step 3: Feature Expansion
        feat = self.nodeshuffle(feat)

        # Step 4: Coordinate Generation
        delta = self.mlp_delta(feat)
        pcd_child = self.up_sampler(pcd) + delta


        return pcd_child, feat

class Feedback_RefinementNet(nn.Module):
    def __init__(self, num_p0=512, up_factors=[1], cycle_num = 1,
                 return_all_res=False):
        super(Feedback_RefinementNet, self).__init__()
        self.num_p0 = num_p0


        uppers = []
        len_up = len(up_factors)

        for i, factor in enumerate(up_factors):
            if i > 0:
                pre_up_factor = up_factors[i-1]
            else:
                pre_up_factor = 1
            uppers.append(FBAC_BLOCK(up_factor=factor, cycle_num=cycle_num))

        self.uppers = nn.ModuleList(uppers)
        self.up_factors = up_factors

        self.cycle_num = cycle_num
        print('#Time steps:{}'.format(self.cycle_num))
        self.return_all = return_all_res


    def forward(self, pcd, partial):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []

        # Initialize input
        pcd = pcd.permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)

        pcd = pcd.permute(0, 2, 1).contiguous()


        feat_state = []
        pcd_state = []

        # Unfolding across time steps
        for i in range(self.cycle_num):
            pcd_list = []
            feat_list = []
            for idx, upper in enumerate(self.uppers):
                if i == 0:
                    # if self.fps_samp == 2 and idx > 0:
                    if idx > 0:
                        npoints = pcd.shape[2]
                        pcd = pcd.permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
                        pcd = fps_subsample(torch.cat([pcd, partial], 1), npoints).permute(0, 2, 1).contiguous()

                    pcd, feat = upper(pcd, None, None, i)
                else:
                    # feedback state from t-1 step
                    pcd_next = pcd_state[i-1][idx]
                    feat_next = feat_state[i-1][idx]

                    if idx == 0:
                        # For 0-th FBAC block
                        pcd = pcd_state[i-1][0]
                        pcd = pcd.permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
                        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0).permute(0, 2, 1).contiguous()

                    else:
                        # For i-th FBAC block (i > 0)
                        pcd = pcd_list[idx-1]
                        
                        npoints = pcd_state[i-1][idx-1].shape[2]
                        pcd = pcd.permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
                        pcd = fps_subsample(torch.cat([pcd, partial], 1), npoints).permute(0, 2, 1).contiguous()


                        
                    pcd, feat = upper(pcd, pcd_next, feat_next, i)
                

                pcd_list.append(pcd)
                feat_list.append(feat)

                if self.return_all:
                    arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
                else:
                    if i == self.cycle_num-1:
                        arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

            # Saving present time step states
            pcd_state.append(pcd_list)
            feat_state.append(feat_list)

        return arr_pcd


class Model(nn.Module):
    def __init__(self, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()

        num_pc = 128 
        num_p0 = 512

        num_points = 2048

        self.coarse_net = HGNet(num_pc=num_pc, g_feat_dim=global_feature_size)

        up_factors=[1,2,2]
        
        cycle_num = 3 
        self.refine = Feedback_RefinementNet(num_p0=num_p0, up_factors=up_factors, cycle_num = cycle_num, return_all_res=True)

    def forward(self, x, gt=None, prefix="test"):
        # Coarse generation
        coarse_pcd, _ = self.coarse_net(x)

        # feedback refinement stage
        res_pcds = self.refine(coarse_pcd, x.transpose(2,1).contiguous())

        fine = res_pcds[-1]
        return fine