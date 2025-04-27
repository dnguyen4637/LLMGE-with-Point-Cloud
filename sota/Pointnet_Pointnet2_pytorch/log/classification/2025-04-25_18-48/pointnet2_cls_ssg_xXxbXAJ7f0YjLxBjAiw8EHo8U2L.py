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

def weighted_nll_loss(input, target, weight):
    """
    Calculates the weighted negative log likelihood loss.
    :param input: (FloatTensor) Input tensor of size (N, C) where N is the number of samples and C is the number of classes.
    :param target: (LongTensor) Target tensor of size (N) containing integer values from [0, C - 1].
    :param weight: (FloatTensor) Weight tensor of size (C) containing the weights for each class.
    :return: (FloatTensor) Loss tensor of size (1).
    """
    nll_loss = F.nll_loss(input, target)
    weighted_nll_loss = nll_loss * weight[target]
    return weighted_nll_loss.mean()

def feature_loss(trans_feat, target_feat):
    """
    Calculates the feature loss between the transformed feature tensor and the target feature tensor.
    :param trans_feat: (FloatTensor) Transformed feature tensor of size (N, D) where N is the number of samples and D is the number of features.
    :param target_feat: (FloatTensor) Target feature tensor of size (N, D).
    :return: (FloatTensor) Feature loss tensor of size (1).
    """
    diff = (trans_feat - target_feat).pow(2).sum(dim=1)
    return diff.mean()

class GetLoss(nn.Module):
    def __init__(self):
        super(GetLoss, self).__init__()
        self.weight = torch.tensor([0.5, 1.0, 2.0])  # Example class weights

    def forward(self, pred, target, trans_feat, target_feat):
        loss = weighted_nll_loss(F.log_softmax(pred, dim=1), target, self.weight) + feature_loss(trans_feat, target_feat)
        return loss
# --OPTION--