# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
from torch import nn
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
        self.fc = nn.Linear(mlp[-1], npoint * 3)
        self.group_all = group_all

    def forward(self, xyz, points):
        batch_size = xyz.shape[0]
        new_xyz = torch.cat([xyz.unsqueeze(1)] * self.nsample, dim=1)
        diff = (new_xyz[:, :, None, :] - points[:, None, :, :]).permute(0, 2, 3, 1)
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        idx = dist.argsort(dim=-1)[:, :, :self.nsample]
        dist_sorted = dist[torch.arange(batch_size)[..., None], idx][..., :self.nsample]
        mask = dist_sorted <= self.radius
        idx_keep = idx[:, :, mask]
        points = points[idx_keep]
        xyz = new_xyz[idx_keep]

        if not self.group_all:
            xyz_norm = torch.norm(xyz[:, :, 0], p=2, dim=-1, keepdim=True)
            xyz = xyz / xyz_norm
            xyz = torch.cat([xyz[:, :, 0].unsqueeze(-1), xyz[:, :, 1].unsqueeze(-1), xyz[:, :, 2].unsqueeze(-1)], dim=-1)

        x = points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs[:-1]):
            x = F.relu(self.mlp_bns[i](conv(x)))
        x = self.mlp_bns[-1](self.mlp_convs[-1](x))

        if self.group_all:
            x = x.mean(dim=2, keepdim=True)
        else:
            x = x.sum(dim=2, keepdim=True)

        x = x.squeeze(-1)
        x = F.relu(self
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
