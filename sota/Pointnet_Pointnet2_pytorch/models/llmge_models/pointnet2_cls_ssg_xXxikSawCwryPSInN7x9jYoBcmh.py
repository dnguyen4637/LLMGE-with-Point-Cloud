# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points

# --OPTION--
import torch.nn.functional as F
import math

class AttentionWeights(nn.Module):
    def __init__(self, in_features):
        super(AttentionWeights, self).__init__()
        self.query = nn.Linear(in_features, 1)

    def forward(self, x):
        attn_weights = self.query(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        return attn_weights

class DynamicLossFactor(nn.Module):
    def __init__(self, warmup_steps):
        super(DynamicLossFactor, self).__init__()
        self.warmup_steps = warmup_steps

    def forward(self, epoch):
        if epoch < self.warmup_steps:
            factor = math.pow(epoch / self.warmup_steps, 0.5)
        else:
            factor = 1.0
        return factor

class get_loss(nn.Module):
    def __init__(self, in_features, warmup_steps):
        super(get_loss, self).__init__()
        self.attn_weights = AttentionWeights(in_features)
        self.dynamic_factor = DynamicLossFactor(warmup_steps)

    def forward(self, pred, target, trans_feat):
        attn_weights = self.attn_weights(trans_feat)
        weighted_pred = pred * attn_weights
        total_loss = F.nll_loss(weighted_pred, target)
        epoch =... # Get current epoch from training loop
        factor = self.dynamic_factor(epoch)
        total_loss *= factor

        return total_loss
# --OPTION--