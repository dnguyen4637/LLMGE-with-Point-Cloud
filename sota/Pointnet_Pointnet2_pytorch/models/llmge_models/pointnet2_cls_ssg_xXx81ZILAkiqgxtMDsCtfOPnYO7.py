# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F

def pointnet_conv(in_channel, out_channel, kernel_size, padding, stride):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channel),
        nn.ReLU()
    )

def pointnet_mlp(in_channel, out_channel):
    return nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.Linear(out_channel, out_channel),
        nn.BatchNorm1d(out_channel),
        nn.ReLU()
    )
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--