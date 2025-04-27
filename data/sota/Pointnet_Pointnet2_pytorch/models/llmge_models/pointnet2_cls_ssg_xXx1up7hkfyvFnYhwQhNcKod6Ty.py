# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
import torch.optim as optim
import math

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.convs = []
        self.bns = []

        for i in range(num_layers):
            conv = nn.Conv1d(input_channels + i * growth_rate, growth_rate, kernel_size=1)
            bn = nn.BatchNorm1d(growth_rate)
            self.convs.append(conv)
            self.bns.append(bn)

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
            x = torch.cat((x, conv(x)), dim=1)

        return x

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[128, 128, 256], group_all=False)
        self.db1 = DenseBlock(num_layers=4, input_channels=256, growth_rate=128)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=384, mlp=[256, 256, 512], group_all=False)
        self.db2 = DenseBlock(num_layers=4, input_channels=768, growth_rate=256)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024, mlp=[512, 1024, 2048], group_all=True)
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--