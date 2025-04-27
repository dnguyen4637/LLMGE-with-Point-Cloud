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

class get_loss(nn.Module):
    def __init__(self, weight_decay=0.0001, label_smoothing=0.1):
        super(get_loss, self).__init__()
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, trans_feat):
        # Calculate the number of classes
        num_classes = pred.shape[1]

        # Calculate the smoothed targets
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smoothed_targets = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (num_classes - 1)

        # Add weight decay to the prediction
        weight_decay_term = torch.tensor(0.).to(pred.device)
        for name, param in pred.named_parameters():
            if 'bias' not in name:
                weight_decay_term += torch.sum(torch.square(param))
        weight_decay_term *= self.weight_decay

        # Calculate the loss
        total_loss = F.cross_entropy(pred, smoothed_targets, reduction='mean') + weight_decay_term

        return total_loss
# --OPTION--