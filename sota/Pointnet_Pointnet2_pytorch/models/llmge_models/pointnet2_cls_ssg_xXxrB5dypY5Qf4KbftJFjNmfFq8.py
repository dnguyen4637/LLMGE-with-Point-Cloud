# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.Sequential(*[nn.Conv1d(in_channel if i == 0 else mlp[i-1], mlp[i], 1) for i in range(len(mlp))])
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.transpose(1, 2) # (batch_size, num_points, 3)
        xyz_new = nn.functional.furthest_point_sampling(xyz, self.npoint) # (batch_size, npoint, 3)
        idx = nn.functional.batch_triangle_sample(xyz, xyz_new) # (batch_size, npoint, nsample)
        grouped_xyz = xyz.new_zeros(xyz.size(0), self.npoint, self.nsample, 3) # (batch_size, npoint, nsample, 3)
        grouped_points = points.new_zeros(points.size(0), self.npoint, self.nsample, points.size(-1)) # (batch_size, npoint, nsample, C)
        for i in range(self.npoint):
            grouped_xyz[..., i*self.nsample:(i+1)*self.nsample,...] = xyz[idx[i]] # (nsample, 3)
            grouped_points[..., i*self.nsample:(i+1)*self.nsample,...] = points[idx[i]] # (nsample, C)
        xyz_new = xyz_new.unsqueeze(-1).expand(-1, -1, -1, points.size(-1)) # (batch_size, npoint, 3, C)
        grouped_points = torch.cat([grouped_points, xyz_new], dim=-1) # (batch_size, npoint, nsample, C+3)
        x = self.mlp_convs(grouped_points) # (batch_size, npoint, nsample, mlp[-1])
        if self.group_all:
            x, _ = torch.max(x, dim=2) # (batch_size, npoint, mlp[-1])
        return xyz_new, x

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--