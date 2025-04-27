# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_cluster import fps

# Helper functions
def fps_helper(xyz, k):
    """Perform farthest point sampling."""
    device = xyz.device
    idx = fps(xyz, k, batch_size=xyz.shape[0])
    return idx

def feature_propagation(xyz, points, idx):
    """Propagate features using farthest point sampling indices."""
    xyz_new = xyz[idx]
    points_new = points.gather(1, idx.unsqueeze(2)).squeeze(-1)
    return xyz_new, points_new

def mlp(x, in_channels, hidden_channels, out_channels, nb_layers, dropout=0.):
    """Apply multi-layer perceptron."""
    for i in range(nb_layers - 1):
        x = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )(x)
        in_channels = hidden_channels
    x = nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )(x)
    return x

class SpatialTransformerNetwork(nn.Module):
    """Spatial Transformer Network."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 64, kernel_size=1)
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 6)
        )

    def forward(self, xyz):
        xyz_conv = self.conv(xyz.transpose(1, 2))
        xyz_conv = xyz_conv.transpose(1, 2)
        loc = self.fc_loc(pad_sequence(torch.unbind(xyz_conv, dim=1), batch_first=True))
        trans = nn.functional.affine_grid(loc, xyz.size())
        xy
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np
