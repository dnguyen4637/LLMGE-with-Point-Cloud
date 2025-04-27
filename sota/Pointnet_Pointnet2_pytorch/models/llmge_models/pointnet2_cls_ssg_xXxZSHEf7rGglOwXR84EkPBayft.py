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
from torch.nn import functional as F

class GetLoss(nn.Module):
    def __init__(self, weight=None, gp_weight=10):
        super(GetLoss, self).__init__()
        self.gp_weight = gp_weight
        if weight is not None:
            self.criterion = nn.NLLLoss(weight=weight)
        else:
            self.criterion = nn.NLLLoss()

    def forward(self, pred, target, trans_feat):
        total_loss = self.criterion(pred, target)

        # Gradient Penalty Term (for GAN-like architectures)
        if len(trans_feat.shape) > 2:
            batch_size = trans_feat.shape[0]
            feat_dim = trans_feat.shape[1]
            epsilon = torch.rand((batch_size, 1, feat_dim)).to(pred.device)
            interpolated = epsilon * pred + ((1 - epsilon) * trans_feat)
            with torch.no_grad():
                disc_interpolated = self.discriminator(interpolated)
            gradients = torch.autograd.grad(outputs=disc_interpolated, inputs=interpolated,
                                             grad_outputs=torch.ones_like(disc_interpolated),
                                             create_graph=True, retain_graph=True)[0]
            gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_weight
            total_loss += gradients_penalty

        return total_loss
# --OPTION--