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
from typing import Tuple

class GetLoss(nn.Module):
    def __init__(self):
        super(GetLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross entropy loss between predicted and actual values.

        Args:
            pred (torch.Tensor): Predicted values from the model.
            target (torch.Tensor): Ground truth labels.

        Returns:
            Loss value.
        """
        return self.criterion(pred, target)

def extract_features(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from predicted values.

    Args:
        pred (torch.Tensor): Predicted values from the model.

    Returns:
        Logits and probabilities.
    """
    logits = pred[:, :, :-1]
    probabilities = torch.softmax(logits, dim=-1)
    return logits, probabilities

# Example usage
model = GetLoss()
pred = torch.randn(32, 10, 10)
target = torch.randint(low=0, high=10, size=(32,))

logits, probabilities = extract_features(pred)
loss = model(pred, target)
print("Logits:", logits.shape)
print("Probabilities:", probabilities.shape)
print("Loss:", loss)
# --OPTION--