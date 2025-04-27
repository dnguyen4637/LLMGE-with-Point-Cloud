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
        self.conv1 = nn.Conv1d(self.in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * npoint, mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, points):
        batch_size = xyz.shape[0]
        if self.group_all:
            grouped_xyz = xyz.unsqueeze(-1).repeat(1, 1, self.nsample)
            grouped_points = points.unsqueeze(-1).repeat(1, 1, self.nsample)
            mask = grouped_points.new_ones((batch_size, self.nsample))
        else:
            dists = ((xyz.unsqueeze(2) - grouped_xyz.transpose(1, 2)) ** 2).sum(dim=-1)
            mask = (dists < self.radius ** 2).float()
            idx = mask.nonzero(as_tuple=False).squeeze()
            grouped_xyz = grouped_xyz[idx]
            grouped_points = grouped_points[idx]
            mask = mask[idx]

        if self.nsample > 0 and self.training:
            fps_idx = knn_point(grouped_xyz, mask, self.npoint)
            grouped_xyz = grouped_xyz[fps_idx]
            grouped_points = grouped_points[fps_idx]
            mask = mask[fps_idx]

        xyz = xyz.unsqueeze(-1).repeat(1, 1, self.nsample)
        grouped_xyz = grouped_xyz - xyz
        x = torch.cat([xyz, grouped_points], dim=-1)
        x = x.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs[:-1]):
            x = F.relu(self.mlp_bns[i](conv(x)))
        x = self.bn2(self.conv2(F.relu(self.bn1(self.mlp_convs[-1](x)))))
        pointfeat = x.permute(0, 2, 1).contiguous()
        x = torch.max(
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--