# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_batch_stats, pointnet2_sa_module, pointnet2_fp_module

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = pointnet2_sa_module(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.fp1 = pointnet2_fp_module(mlp=[128])
        self.sa2 = pointnet2_sa_module(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.fp2 = pointnet2_fp_module(mlp=[256])
        self.sa3 = pointnet2_sa_module(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = pointnet2_fp_module(mlp=[512, 1024])
        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.conv3 = nn.Conv1d(256, num_class, 1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = self.fp1(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.fp2(l2_xyz, l2_points)
        l3_xyz, l3_points = self.sa3(l2_
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--