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
            grouped_xyz = xyz.unsqueeze(0).repeat(batch_size, 1, 1)
            grouped_points = points.unsqueeze(0).repeat(batch_size, 1, 1)

        xyz_reshaped = grouped_xyz.permute(0, 2, 1).contiguous()
        pointcloud_feature = grouped_points.permute(0, 2, 1).contiguous()

        for i, conv in enumerate(self.mlp_convs):
            feature = F.relu(self.mlp_bns[i](conv(pointcloud_feature)))
            pointcloud_feature = feature

        if self.group_all:
            trans_feat = pointcloud_feature.mean(dim=2, keepdim=True)
        else:
            trans_feat = torch.cat([pointcloud_feature.mean(dim=2, keepdim=True),
                                   torch.mean(torch.square(pointcloud_feature - trans_feat.detach()), dim=2, keepdim=True) -
                                   torch.square(trans_feat.detach()).sum(dim=2, keepdim=True) + 1e-6], dim=-1)

        return new_xyz, trans_feat
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--