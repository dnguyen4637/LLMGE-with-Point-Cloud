# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList([nn.Conv1d(in_channel, mlp[i], 1) for i in range(len(mlp))])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm1d(mlp[i]) for i in range(len(mlp))])
        self.group_all = group_all

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        xyz_new = xyz[:, :3, :].contiguous()
        if self.group_all:
            idx = torch.arange(N).long().to(device)
            repeat_times = self.npoint // N
            idx_repeat = idx.repeat(repeat_times, 1)
            xyz_new = xyz_new.repeat(repeat_times, 1, 1)
            points = points.repeat(repeat_times, 1, 1)
            N_new = self.npoint
        else:
            idx = torch.from_numpy(np.random.choice(N, size=self.npoint, replace=True)).long().to(device)
            N_new = self.npoint

        dists = ((xyz_new[:, None, :, :3] - xyz[:, :, idx][:, None, :, :3]).pow(2).sum(-1)).sqrt()
        mask = dists < self.radius
        dists_masked = dists * mask
        idx_knn = dists_masked.topk(k=self.nsample, dim=-1, largest=False)[1]
        xyz_knn = xyz_new[:, :, idx_knn]
        points_knn = points[:, :, idx_knn]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points_knn = conv(points_knn)
            points_knn = F.relu(bn(points_knn))

        if self.group_all:
            fps_idx = torch.arange(N_new).long().to(device)
            points_knn = points_knn.transpose(1, 2).contiguous()
            xyz_knn = xyz_knn.transpose(1, 2).contiguous()
            return xyz_knn, points_knn
        else:
            return xyz[:, :, idx], points_knn

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--