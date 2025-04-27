# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn

class TransitionDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransitionDown, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = torch.max(x, dim=-1, keepdim=True)[0]
        return x
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--