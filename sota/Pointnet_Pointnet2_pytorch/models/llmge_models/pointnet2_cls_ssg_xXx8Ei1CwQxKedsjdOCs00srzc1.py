# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
def __init__(...):
   ...
    self.conv1 = torch.nn.Conv1d(in_channels=in_channel, out_channels=64)
   ...

def forward(self, xyz):
   ...
    if self.normal_channel:
        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]
    else:
        norm = None
    xyz = self.conv1(xyz.transpose(-1, -2))  # Transpose for Conv1d
    xyz = torch.nn.functional.batch_norm(xyz, momentum=0.01)
    xyz = torch.nn.functional.relu(xyz)
    l1_xyz, l1_points = self.sa1(xyz, norm)
   ...
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--