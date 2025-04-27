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

class GetLoss(nn.Module):
    def __init__(self, epsilon=1e-8, reduction='mean', alpha=0.99):
        super(GetLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, pred, target, trans_feat):
        # Calculate the mean over non-padding elements
        batch_size, _, _ = pred.shape
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        # Apply label smoothing
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one = (1 - self.epsilon) / (n_classes - 1)
        smooth_zero = self.epsilon / n_classes
        one_hot = one_hot * smooth_zero + (1 - one_hot) * smooth_one

        # Calculate cross entropy loss
        ce_loss = -torch.sum(one_hot * F.log_softmax(pred, dim=-1), dim=-1)

        # Add feature-based regularization term
        reg_term = torch.mean(trans_feat ** 2)

        # Combine losses
        total_loss = ce_loss + self.alpha * reg_term

        if self.reduction =='mean':
            total_loss = total_loss.mean()
        elif self.reduction =='sum':
            total_loss = total_loss.sum()

        return total_loss
# --OPTION--