# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
from torch import nn
from torch.nn import functional as F

class MultiRadiusPointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlps, group_alls):
        super().__init__()
        self.mlp_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(in_channel, mlp, kernel_size=1),
                nn.BatchNorm1d(mlp),
                nn.ReLU(),
                nn.Conv1d(mlp, mlp, kernel_size=1),
                nn.BatchNorm1d(mlp),
                nn.ReLU()
            )
            for mlp in mlps
        ])
        self.sa_modules = nn.ModuleList([
            PointNetSetAbstraction(npoint, radius, nsample, in_channel, [mlp], group_all)
            for radius, nsample, mlp in zip(radii, nsamples, mlps)
        ])

    def forward(self, xyz, norm):
        for sa_module in self.sa_modules:
            xyz, points = sa_module(xyz, norm)
        x = self.mlp_blocks(points.transpose(1, 2)).transpose(1, 2)
        return xyz, x

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa = MultiRadiusPointNetSetAbstraction(
            npoint=[512, 128, None],
            radii=[0.2, 0.4, None],
            nsamples=[32, 64, None],
            in_channel=in_channel,
            mlps=[64, 128, 256],
            group_alls=[False, False, True]
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--