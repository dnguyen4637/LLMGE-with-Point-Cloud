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
import torch.nn as nn
import torch.nn.functional as F

class GetLossWithSpatialWeightingAndRegularization(nn.Module):
    def __init__(self, alpha=0.5, beta=0.0001):
        super(GetLossWithSpatialWeightingAndRegularization, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target, trans_feat):
        # Calculate spatial weighting based on the feature map
        spatial_weight = F.adaptive_avg_pool2d(trans_feat, output_size=(1, 1))
        spatial_weight = spatial_weight.view(-1, 1, 1, 1)
        spatial_weight = F.softmax(spatial_weight, dim=1)

        # Apply spatial weighting to the cross-entropy loss
        total_loss = F.nll_loss(pred * spatial_weight, target)

        # Add L2 regularization term
        l2_regularization = self.beta * sum(torch.square(param) for param in self.parameters())

        # Combine the losses
        final_loss = total_loss + l2_regularization

        return final_loss
# --OPTION--