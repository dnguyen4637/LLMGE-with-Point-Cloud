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
        self.fc1 = nn.Linear(mlp[0], mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, points):
        batch_size = xyz.shape[0]
        new_xyz = nn.functional.farthest_point_sampling(xyz, self.npoint)
        if self.group_all:
            idx = torch.arange(batch_size, device=xyz.device).repeat(self.npoint, 1).t().contiguous().view(-1)
            new_points = points.view(batch_size, -1, self.in_channel)[idx, :].view(batch_size * self.npoint, self.in_channel)
        else:
            idx = torch.arange(batch_size, device=xyz.device).repeat(self.npoint, 1)
            dists = ((new_xyz[:, None, :] - xyz[idx, None]) ** 2).sum(dim=-1)
            dists, idx = dists.sort(dim=-1, descending=False)
            idx = idx[:, :self.nsample]
            new_points = points[idx]
            new_points = new_points.view(batch_size, self.nsample * self.npoint, self.in_channel)

        x = new_points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs[:-1]):
            x = F.relu(self.mlp_bns[i](conv(x)))
        x = self.fc1(F.relu(self.bn1(self.conv1(x))))
        x = x.mean(dim=2)
        return new_xyz, x

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.
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
