# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    # Unchanged PointNetSetAbstraction implementation
    pass

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=1)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(512, num_class, kernel_size=1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        x = self.conv1(xyz.transpose(1, 2))
        l1_xyz, l1_points = self.sa1(x, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = self.gap(l3_points.transpose(1, 2)).squeeze(-1)
        x = self.drop1(F.relu(self.bn1(self.conv2(x))))
        x = self.conv3(x)
        x = F.log_softmax
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--