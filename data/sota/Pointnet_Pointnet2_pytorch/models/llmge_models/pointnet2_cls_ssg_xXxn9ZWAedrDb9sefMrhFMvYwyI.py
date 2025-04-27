# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class PointNetSetAbstraction(nn.Module):
    # Original implementation remains unchanged

class PointwiseAttention(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(gating_channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv1d(inter_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        avg_pool = nn.AdaptiveAvgPool1d(1)(x)
        max_pool = nn.AdaptiveMaxPool1d(1)(x)

        gate = self.conv2(torch.cat([avg_pool, max_pool], dim=-1))
        feature = self.conv1(x)

        return self.conv3(gate * feature)

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.att1 = PointwiseAttention(128, 64, 64)

        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.att2 = PointwiseAttention(256, 128, 128)

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.att3 = PointwiseAttention(1024, 256, 256)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--