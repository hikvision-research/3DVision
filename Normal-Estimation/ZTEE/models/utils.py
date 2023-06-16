#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, grouping_operation





def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

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
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
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
    return group_idx.int()


class EdgeConv(torch.nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C2, N]
    """

    def __init__(self, input_channel, output_channel, k=20 ):
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
            neigh_feature = group_local(inputs, k=self.num_neigh)
            central_feat = inputs.unsqueeze(dim=3).repeat(1, 1, 1, self.num_neigh)
        else:
            central_feat = torch.zeros(batch_size, dims, num_points, 1).to(inputs.device)
            neigh_feature = inputs.unsqueeze(-1)
        edge_feature = central_feat - neigh_feature.contiguous()
        feature = torch.cat((edge_feature, central_feat), dim=1)
        feature = self.conv(feature)
        central_feature = feature.max(dim=-1, keepdim=False)[0]
        return central_feature

def group_local(xyz, k=20, return_idx=False):
    """
    Input:
        x: point cloud, [B, 3, N]
    Return:
        group_xyz: [B, 3, N, K]
    """
    xyz = xyz.transpose(2, 1).contiguous()
    idx = query_knn_point(k, xyz, xyz)
    # torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx.long())
    # torch.cuda.empty_cache()
    group_xyz = group_xyz.permute(0, 3, 1, 2)
    if return_idx:
        return group_xyz, idx

    return group_xyz


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



class AdaptGraphPooling_with_npoints(nn.Module):
    def __init__(self, in_channel, num_samples, neighbor_num=8, dim=64, center_fixed=False):
        super().__init__()
        self.num_samples = num_samples
        self.neighbor_num = neighbor_num
        self.center_fixed = center_fixed
        # print('Center_fixed: {}'.format(center_fixed))

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(dim, in_channel, 1)
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
        new_npoints = self.num_samples
        key_points_idx = furthest_point_sample(vertices.transpose(2,1).contiguous(), new_npoints)
        if self.center_fixed:
            key_points_idx[:,0] = 0

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
  
        group_feat = group_feat + pos_embedding  #
        new_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)

        return key_point, new_feat
