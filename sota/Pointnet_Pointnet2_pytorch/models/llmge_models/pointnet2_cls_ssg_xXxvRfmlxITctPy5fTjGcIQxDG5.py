# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from pointnet2_modules import PointNetSetAbstraction

class SimplifiedModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(SimplifiedModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=in_channel, mlp=[128, 128, 256], group_all=True)
        self.fc1 = nn.Linear(256 + 3, 128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz, points = self.sa(xyz, norm)
        x = torch.cat([xyz.mean(dim=1, keepdim=True), points], dim=-1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, -1)

        return x, points
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--