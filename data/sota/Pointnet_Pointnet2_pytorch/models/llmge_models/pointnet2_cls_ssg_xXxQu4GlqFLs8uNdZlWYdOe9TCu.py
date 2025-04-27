# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from pointnet2_modules import PointNetSetAbstraction

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc = self.get_mlp([1024, 512, 256, num_class])
        self.norm_dropout = self.get_norm_dropout(512, 0.4)
        self.norm_dropout_2 = self.get_norm_dropout(256, 0.4)

    @staticmethod
    def get_mlp(output_dims):
        layers = []
        for i in range(len(output_dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(output_dims[i], output_dims[i+1]),
                nn.ReLU()
            ))
        return nn.Sequential(*layers)

    @staticmethod
    def get_norm_dropout(num_features, p):
        return nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def extract_features(self, xyz, norm):
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(xyz.size(0), -1)
        return x

    def forward(self, xyz):
        if self.normal_channel:
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--