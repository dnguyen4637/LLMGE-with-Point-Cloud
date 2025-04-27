# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList([nn.Conv1d(in_channel, mlp[i], 1) for i in range(len(mlp))])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm1d(mlp[i]) for i in range(len(mlp))])
        self.group_all = group_all

    def forward(self, xyz, points):
        batch_size = xyz.shape[0]
        new_xyz = nn.functional.furthest_point_sampling(xyz, self.npoint)
        if not self.group_all:
            idx = nn.functional.batch_triangle_sample(new_xyz, xyz, self.nsample)
            grouped_xyz = xyz[idx]
            grouped_points = points[idx]
        else:
            grouped_xyz = xyz.unsqueeze(1).repeat(1, self.nsample, 1)
            grouped_points = points.unsqueeze(1).repeat(1, self.nsample, 1)

        xyz_reshaped = grouped_xyz.permute(0, 2, 1).contiguous()
        pointcloud_feature = grouped_points
        for i, conv in enumerate(self.mlp_convs):
            feature = conv(pointcloud_feature)
            feature = F.relu(self.mlp_bns[i](feature))
            pointcloud_feature = feature

        if self.group_all:
            trans_vec = nn.functional.linear(pointcloud_feature, 3 * self.npoint)
            trans_vec = trans_vec.permute(0, 2, 1).contiguous().view(-1, 3)
            new_xyz = new_xyz.view(-1, 3)
            diff = new_xyz[:, None, :] - trans_vec[None, :, :]
            pairwise_distance = torch.pow(diff, 2).sum(dim=-1)
            idx = pairwise_distance.argsort(dim=-1)[:, :self.npoint]
            new_xyz = new_xyz[idx]
            pointcloud_feature = pointcloud_feature.view(batch_size, -1, self.npoint)[idx, :, :].view(batch_size, self.npoint, -1)

        return new_xyz, pointcloud_feature
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--