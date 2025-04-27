# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.dataset import ShapeNetDataset

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, nb_layers):
        super(DenseBlock, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=1) for i in range(nb_layers)])
        self.norm = nn.BatchNorm1d(in_channels + nb_layers * growth_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(torch.cat([x, out], dim=1))
        return out

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.db1 = DenseBlock(128, 64, 4)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=192, mlp=[128, 128, 256], group_all=False)
        self.db2 = DenseBlock(256, 128, 4)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=384 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--