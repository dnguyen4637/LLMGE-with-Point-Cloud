# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class WeightedPointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channel, mlp[i], 1),
                                                      nn.BatchNorm1d(mlp[i]),
                                                      nn.ReLU()) for i in range(len(mlp))])
        self.group_all = group_all

    def forward(self, xyz, points):
        device = xyz.device
        batch_size = xyz.shape[0]
        coords, idx = self._get_coordinates(xyz)

        # Sample points within the given radius
        k_nearest = min(self.nsample, coords.shape[1])
        _, idx = torch.topk(coords, k_nearest, dim=1, largest=False, sorted=False)
        _, new_idx = torch.sort(idx, dim=1)
        new_xyz = xyz[:, new_idx, :]
        new_points = points[:, new_idx, :]

        # Calculate distances between sampled points and centroids
        dists = torch.norm(new_xyz[:, :, None, :] - new_xyz[:, None, :, :], dim=-1)
        max_dists = torch.clamp(dists, min=1e-10)

        # Apply Gaussian weights to points
        gaussian_weights = torch.exp(-dists / (2 * max_dists))

        # Normalize weights per point
        normalized_weights = gaussian_weights / torch.sum(gaussian_weights, dim=-2, keepdim=True)

        # Multiply points by their corresponding weights
        weighted_points = torch.sum(new_points * normalized_weights.unsqueeze(-1), dim=-2)

        # Pass data through MLP layers
        for mlp_conv in self.mlp_convs:
            weighted_points = mlp_conv(weighted_points)

        # Group all points together or use the specified number of points
        if self.group_all:
            xyz_out = torch.mean(new_xyz, dim=1, keepdim=True)
            points_out = weighted_points
        else:
            xyz_out = xyz[:, :self.npoint, :]
            points_out = weighted_points[:self.npoint, :]

        return xyz_out, points_out

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
