# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        avg_pool = nn.AdaptiveAvgPool1d(1)(x)
        max_pool = nn.AdaptiveMaxPool1d(1)(x)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attn_map = self.conv1(pooled)
        attn_map = self.softmax(attn_map)
        weighted = x * attn_map
        return weighted

class ResidualConnection(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualConnection, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x):
        skip = x
        x = self.fc(x)
        x += skip
        return x

class get_model(nn.Module):
    #... (same as before)

    def forward(self, xyz):
        #... (same as before)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = SpatialAttention(512)(x)
        x = ResidualConnection(512, 512)(x)
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = SpatialAttention(256)(x)
        x = ResidualConnection(256, 256)(x)
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points

#... (training loop using DataLoader, EarlyStopping, etc.)
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
