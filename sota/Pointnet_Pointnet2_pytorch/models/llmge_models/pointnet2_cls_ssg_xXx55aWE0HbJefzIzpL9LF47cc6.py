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
        B, C, N = xyz.shape
        if self.group_all:
            idx = torch.arange(N).unsqueeze(0).repeat(B, 1)
            grouped_xyz = xyz[:, :, idx].contiguous().view(B, C, 1, N)
            grouped_points = points.transpose(2, 1).contiguous().view(B, self.in_channel, 1, N)
        else:
            farthest_points = self.furthest_point_sample(xyz, self.nsample)
            grouped_xyz = xyz[:, :, farthest_points].contiguous().view(B, C, 1, self.nsample)
            grouped_points = points.transpose(2, 1).contiguous().view(B, self.in_channel, self.nsample, 1)

        trans_grouped_points = grouped_points - grouped_xyz.transpose(2, 3)
        trans_grouped_points = torch.cat([grouped_xyz.transpose(2, 3), trans_grouped_points], dim=1)

        for i, conv in enumerate(self.mlp_convs):
            trans_grouped_points = F.relu(self.mlp_bns[i](conv(trans_grouped_points)))

        new_xyz = grouped_xyz.squeeze(3)
        new_points = trans_grouped_points.mean(dim=2)

        return new_xyz, new_points

    @staticmethod
    def furthest_point_sample(xyz, npoint):
        device = xyz.device
        B, C, N = xyz.shape
        idx = torch.arange(N, device=device).repeat(B, 1)
        distance, idx_dis = torch.topk(torch.cdist(xyz, xyz, p=2).view(B, N, N), k=npoint, dim=-1, largest=True, sorted=False)
        idx_dis = idx_dis.view(B, npoint, 1)
        idx = idx.view(B, 1, N).
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--