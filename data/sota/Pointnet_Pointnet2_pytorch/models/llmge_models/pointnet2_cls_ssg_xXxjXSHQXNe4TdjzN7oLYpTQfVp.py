# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from pointnet2_modules import PointNetSetAbstractionMsg

def create_mlp(channels):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Sequential(
            nn.Conv1d(channels[i], channels[i+1], 1),
            nn.BatchNorm1d(channels[i+1]),
            nn.ReLU(),
        ))
    return nn.Sequential(*layers)

def create_setabstraction(npoint, radius, nsample, in_channel, mlp, group_all):
    return PointNetSetAbstractionMsg(npoint, radius, nsample, in_channel, mlp, group_all)

def create_fc_block(in_features, hidden_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.LogSoftmax(-1)
    )

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = create_setabstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = create_setabstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = create_setabstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.mlp = create_mlp([1024, 512, 256, num_class])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points =
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--