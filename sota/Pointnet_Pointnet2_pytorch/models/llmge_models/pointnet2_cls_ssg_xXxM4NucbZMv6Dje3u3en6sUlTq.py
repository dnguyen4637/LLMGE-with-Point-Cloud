
# ========== Start: GeneCrossed

# ========== End:

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        #... (rest of the code remains unchanged)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        #... (rest of the code remains unchanged)
# --OPTION--
import torch
import torch.nn as nn
from models.pointnet_utils import PointNetSetAbstraction

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa12 = PointNetSetAbstraction(npoint=128, radius=[0.2, 0.4], nsample=[32, 64], in_channel=in_channel, mlp=[64, 64, 128], group_all=[False, False])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc = nn.Linear(1024, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l12_xyz, l12_points = self.sa12(xyz, norm)
        l3_xyz, l3_points = self.sa3(l12_xyz, l12_points)
        x = l3_points.view(B, 1024)
        x = self.fc(x)
        x = torch.log_softmax(x, dim=-1)

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
