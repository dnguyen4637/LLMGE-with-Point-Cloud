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
import torch
import torch.nn as nn
import torch.nn.functional as F

class get_loss(nn.Module):
    def __init__(self, num_classes, l2_lambda=0.01):
        super(get_loss, self).__init__()
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda

        # Weighted Cross Entropy Loss
        self.weights = torch.ones(num_classes)
        self.weights[0] *= 0.5  # Adjust the weight of class 0

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(trans_feat.shape[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, pred, target, trans_feat):
        batch_size = pred.shape[0]

        # Weighted Cross Entropy Loss
        weighted_cross_entropy = -torch.sum(F.log_softmax(pred, dim=-1) * target, dim=-1)
        weighted_cross_entropy = torch.mean(weighted_cross_entropy * self.weights)

        # Attention Mechanism
        attn_weights = self.attention(trans_feat)
        attn_pred = pred * attn_weights.unsqueeze(-1)

        # L2 Regularization
        l2_regularization = torch.tensor(0., device=pred.device)
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l2_regularization += torch.sum(torch.square(param))

        total_loss = weighted_cross_entropy + self.l2_lambda * l2_regularization

        return total_loss
# --OPTION--