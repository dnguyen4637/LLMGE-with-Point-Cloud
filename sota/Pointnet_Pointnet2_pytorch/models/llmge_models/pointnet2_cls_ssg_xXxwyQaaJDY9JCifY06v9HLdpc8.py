# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
from pointnet2_ops import PointNetSetAbstractionMsg

def create_sa_layer(npoint, radius, nsample, in_channel, mlp, group_all):
    return PointNetSetAbstractionMsg(npoint=npoint, radius=radius, nsample=nsample, in_channel=in_channel, mlp=mlp, group_all=group_all)

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        
        self.sa1 = create_sa_layer(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = create_sa_layer(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = create_sa_layer(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_class),
            nn.LogSoftmax(-1)
        )

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
        x = self.fc(x)
        
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
