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
        self.sa = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[256, 256, 256], group_all=False)
        self.fc1 = nn.Linear(256 * 2 + 3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa(xyz, norm)
        x = torch.cat([l1_xyz, l1_points], dim=-1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        x = F.log_softmax(x, -1)

        return x, l1_points
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--