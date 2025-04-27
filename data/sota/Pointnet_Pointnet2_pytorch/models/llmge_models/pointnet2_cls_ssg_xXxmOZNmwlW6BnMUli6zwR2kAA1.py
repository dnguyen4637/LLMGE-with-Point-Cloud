# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    #... (same as original implementation)

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 768)  # Changed from 512 to 768
        self.bn1 = nn.BatchNorm1d(768)   # Adjusted batchnorm accordingly
        self.drop1 = nn.Dropout(0.6)     # Increased dropout probability
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(768, 512)   # Changed from 256 to 512
        self.bn2 = nn.BatchNorm1d(512)   # Adjusted batchnorm accordingly
        self.drop2 = nn.Dropout(0.6)     # Increased dropout probability
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(512, num_class)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

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
        x = self.drop1(self.leaky_rel
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
