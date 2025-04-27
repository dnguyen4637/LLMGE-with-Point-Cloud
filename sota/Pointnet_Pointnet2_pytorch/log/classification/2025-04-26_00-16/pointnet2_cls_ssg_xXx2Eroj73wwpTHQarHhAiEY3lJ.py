# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiScalePointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlps, group_all=False):
        super().__init__()
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList([
            PointNetConv(in_channel, mlp[0], radii=r, nsample=ns, group_all=group_all)
            for r, ns, mlp in zip(radii, nsamples, mlps)
        ])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm1d(m) for m in mlps[0][:-1]])
        self.mlp_drops = nn.ModuleList([nn.Dropout(0.4) for _ in range(len(mlps)-1)])

    def forward(self, xyz, points):
        for i, (conv, bn, drop) in enumerate(zip(self.mlp_convs, self.mlp_bns, self.mlp_drops)):
            xyz, points = conv(xyz, points)
            points = bn(points)
            points = F.relu(points)
            if i < len(self.mlp_drops):
                points = drop(points)
        return xyz, points

class AttentionModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, _ = x.shape
        query = self.query_conv(x).view(batch_size, -1, self.query_conv.out_channels)
        key = self.key_conv(x).view(batch_size, -1, self.key_conv.out_channels)
        value = self.value_conv(x).view(batch_size, -1, self.value_conv.out_channels)

        attn_score = torch.matmul(query, key.transpose(-1, -2))
        attn_score = F.softmax(attn_score, dim=-1)

        attended_value = torch.matmul(attn_score, value)
        attended_value = attended_value.view(batch_size, -1, attended_value.shape[-1])

        out = self.gamma * attended_value + x
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
