# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList([nn.Conv1d(in_channel, mlp[i], 1) for i in range(len(mlp))])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm1d(mlp[i]) for i in range(len(mlp))])
        self.conv1 = nn.Conv1d(self.in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * npoint, mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        if self.group_all:
            idx = torch.arange(N).unsqueeze(0).repeat(B, 1)
            new_xyz = xyz[:, :, idx].contiguous().view(B, C, 1)
            new_points = points.transpose(2, 1).contiguous().view(B, self.in_channel, 1)
        else:
            fps_idx = self.furthest_point_sample(xyz, self.nsample)
            new_xyz = xyz[fps_idx, :]
            new_points = points[fps_idx, :]

        trans_points = new_points - new_xyz.transpose(2, 1)
        trans_points = torch.cat([new_xyz.transpose(2, 1), trans_points], dim=-1)

        for i, conv in enumerate(self.mlp_convs[:-1]):
            trans_points = F.relu(self.mlp_bns[i](conv(trans_points)))

        xyz_norm = F.normalize(new_xyz, p=2, dim=-1)
        last_conv = self.mlp_convs[-1]
        last_bn = self.mlp_bns[-1]
        trans_points = F.relu(last_bn(last_conv(trans_points)))

        if self.group_all:
            trans_points = trans_points.view(B, -1, trans_points.size(-1))
            xyz_norm = xyz_norm.view(B, -1, xyz_norm.size(-1)).repeat(1, self.npoint, 1)
        else:
            trans
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--