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
from einops import rearrange

class GetLoss(nn.Module):
    def __init__(self):
        super(GetLoss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # Original NLL Loss
        nll_loss = F.nll_loss(pred, target)

        # New Perceptual Loss
        perceptual_loss = self.perceptual_loss(pred, trans_feat)

        # Weighted Sum of Losses
        total_loss = 0.8 * nll_loss + 0.2 * perceptual_loss

        return total_loss

    def perceptual_loss(self, pred, trans_feat):
        # Calculate intermediate features from the prediction
        pred_features = self.extract_features(pred)

        # Compare the intermediate features with the transformer features
        loss = F.mse_loss(pred_features, trans_feat)

        return loss

    def extract_features(self, x):
        # Define the layers to extract features from the prediction
        layers = [nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU()]

        # Apply each layer sequentially
        x = x.permute(0, 3, 1, 2)  # Change shape from (batch_size, channels, height, width) to (batch_size, height, width, channels)
        for layer in layers:
            x = layer(x)

        # Flatten the output
        x = x.view(x.shape[0], -1)

        return x
# --OPTION--