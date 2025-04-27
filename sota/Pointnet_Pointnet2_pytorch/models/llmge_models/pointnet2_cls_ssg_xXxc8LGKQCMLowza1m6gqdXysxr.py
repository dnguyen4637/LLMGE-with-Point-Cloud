# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from pointnet2_ops import PointNetSetAbstractionMsg

class Model(nn.Module):
    def __init__(self, num_class):
        super(Model, self).__init__()

        # Static MLP parameters
        mlp_params = [
            [[64, 64, 128]],
            [[128, 128, 256]],
            [[256, 512, 1024]]
        ]

        self.sa1 = PointNetSetAbstractionMsg(npoint=512, radius=0.2, nsample=32, in_channel=6 if True else 3, mlp=mlp_params[0][0])
        self.sa2 = PointNetSetAbstractionMsg(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=mlp_params[1][0])
        self.sa3 = PointNetSetAbstractionMsg(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=mlp_params[2][0])

        # Merge fc1, bn1, drop1, fc2, bn2, drop2 into one layer
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, num_class),
            nn.LogSoftmax(-1)
        )

    def forward(self, xyz):
        B, _, _ = xyz.shape

        if xyz.shape[1] == 6:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.fc(
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--