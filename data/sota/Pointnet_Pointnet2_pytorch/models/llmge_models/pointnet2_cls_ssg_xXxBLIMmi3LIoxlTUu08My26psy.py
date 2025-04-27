# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    # Original implementation from https://github.com/fxia22/pointnet.pytorch
    pass

class ModelHelperFunctions:
    @staticmethod
    def process_input(model, xyz, normal_channel=True):
        """
        Processes the input data and prepares it for the model.
        """
        in_channel = 6 if normal_channel else 3
        if normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = model.sa1(xyz, norm)
        l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = model.sa3(l2_xyz, l2_points)
        x = l3_points.view(xyz.size(0), 1024)
        return x, l3_points, l3_xyz

    @staticmethod
    def process_output(model, x, l3_points):
        """
        Post-processes the model output.
        """
        x = model.drop1(F.relu(model.bn1(model.fc1(x))))
        x = model.drop2(F.relu(model.bn2(model.fc2(x))))
        x = model.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x

class GetModel(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(GetModel, self).__init__()
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 if not normal_channel else 6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + (3 if normal_channel else 0), mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + (3 if normal_channel else 0), mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--