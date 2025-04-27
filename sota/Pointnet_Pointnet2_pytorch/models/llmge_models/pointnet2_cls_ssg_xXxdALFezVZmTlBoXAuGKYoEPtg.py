# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class RandomRotationMatrixGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        device = xyz.device
        angle = torch.tensor([torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item()]).to(device)
        q = torch.tensor([torch.sin(angle / 2), torch.cos(angle / 2)]).repeat(3, 1).t()
        R = torch.zeros(3, 3).to(device)
        R[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        R[1, 1] = q[1] ** 2 + q[2] ** 2 - q[0] ** 2 - q[3] ** 2
        R[2, 2] = q[2] ** 2 + q[3] ** 2 - q[0] ** 2 - q[1] ** 2
        R[0, 1] = 2 * (q[0] * q[1] - q[2] * q[3])
        R[0, 2] = 2 * (q[0] * q[2] + q[1] * q[3])
        R[1, 0] = 2 * (q[0] * q[1] + q[2] * q[3])
        R[1, 2] = 2 * (q[1] * q[2] - q[0] * q[3])
        R[2, 0] = 2 * (q[0] * q[2] - q[1] * q[3])
        R[2, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        return xyz @ R

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int], group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList([nn.Conv1d(in_channel, mlp[i], 1) for i in range(len(mlp))])
        self.group_all = group_all
        self.conv1 = nn.Conv1d(mlp[-1], mlp[-1], 1)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = RandomRotationMatrixGenerator()(xyz) # Apply random rotation matrix
        B
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--