# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    # Original implementation from https://github.com/fxia22/pointnet.pytorch
    pass

def create_sa_layer(in_channel, mlp, npoint, radius, nsample, group_all):
    return PointNetSetAbstraction(npoint, radius, nsample, in_channel, mlp, group_all)

def bn_dropout(x, batch_norm, dropout):
    x = batch_norm(x)
    x = dropout(F.relu6(x))
    return x

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = create_sa_layer(in_channel, [64, 64, 128], 512, 0.2, 32, False)
        self.sa2 = create_sa_layer(128 + 3, [128, 128, 256], 128, 0.4, 64, False)
        self.sa3 = create_sa_layer(256 + 3, [256, 512, 1024], None, None, None, True)
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, num_class)

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
        x = bn_dropout(x, self.bn1, self.drop1)
        x = bn_dropout(F.relu6(self.fc1(x)), self.bn2, self.drop2)
        x = self.fc3(
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--