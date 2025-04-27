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

class WeightedNLLLoss(nn.Module):
    def __init__(self):
        super(WeightedNLLLoss, self).__init__()

    def forward(self, pred, target, weights):
        nll_loss = F.nll_loss(pred, target, reduction='none')
        loss = (nll_loss * weights).mean()

        return loss

class GradientPenalty(nn.Module):
    def __init__(self, device):
        super(GradientPenalty, self).__init__()
        self.device = device

    def forward(self, real_data, discriminator):
        alpha = torch.randn(real_data.size(0), 1, 1, 1).to(self.device)
        interpolated = alpha * real_data + ((1 - alpha) * discriminator.fixed_noise.detach())
        interpolated = autograd.Variable(interpolated, requires_grad=True)
        disc_interpolated = discriminator(interpolated)
        gradients = autograd.grad(outputs=disc_interpolated, inputs=interpolated,
                                   grad_outputs=torch.ones(disc_interpolated.size()).to(self.device),
                                   create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.weighted_nll_loss = WeightedNLLLoss()
        self.gradient_penalty = GradientPenalty(device)

    def forward(self, pred, target, trans_feat, real_data, discriminator):
        total_loss = self.weighted_nll_loss(pred, target, trans_feat)
        if 'G_EPOCH' in globals():
            gp = self.gradient_penalty(real_data, discriminator)
            total_loss += gp

        return total_loss
# --OPTION--