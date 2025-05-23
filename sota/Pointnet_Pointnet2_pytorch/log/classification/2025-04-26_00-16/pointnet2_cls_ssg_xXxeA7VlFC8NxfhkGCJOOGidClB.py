# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
import torch.nn.functional as F

def pointnet_sa_module(npoint, radius, nsample, in_channel, mlp, group_all):
    return PointNetSetAbstraction(npoint, radius, nsample, in_channel, mlp, group_all)

def dropout_batchnorm_relu(in_features, dropout_rate):
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.BatchNorm1d(in_features),
        nn.ReLU()
    )

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3

        self.normal_channel = normal_channel

        self.sa1 = pointnet_sa_module(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = pointnet_sa_module(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = pointnet_sa_module(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc1_seq = dropout_batchnorm_relu(512, 0.4)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_seq = dropout_batchnorm_relu(256, 0.4)

        self.fc3 = nn.Linear(256, num_class)
        self.fc3_seq = nn.LogSoftmax(dim=-1)

    def forward(self, xyz):
        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.fc1_seq(self.fc1(x))
        x = self.fc2_seq(self.fc2(x))
        x = self.fc3_seq(self.fc3(x))

        return x, l3_points
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
