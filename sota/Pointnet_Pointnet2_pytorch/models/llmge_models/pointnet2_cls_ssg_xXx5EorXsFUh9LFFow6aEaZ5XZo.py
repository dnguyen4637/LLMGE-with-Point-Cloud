# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
from pointnet2_modules import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.do1 = nn.Dropout(0.4)
        self.conv1 = nn.Conv1d(512, num_class, kernel_size=1)
        self.pool1 = nn.AdaptiveAvgPool1d(1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        sigma1 = nn.Parameter(torch.Tensor(1, 128))
        l1_points = torch.cat([l1_points, torch.abs(l1_points * sigma1.unsqueeze(-1)).mean(dim=-2, keepdim=True)], dim=-1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        sigma2 = nn.Parameter(torch.Tensor(1, 256))
        l2_points = torch.cat([l2_points, torch.abs(l2_points * sigma2.unsqueeze(-1)).mean(dim=-2, keepdim=True)], dim=-1)
        l3_xyz, l
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--