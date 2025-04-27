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
import torch.nn.functional as F
from torchvision import transforms

class get_loss(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super(get_loss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # Weighted Cross Entropy Loss
        self.weights = torch.tensor([0.25, 0.75], dtype=torch.float32).to(device)
        self.weighted_cross_entropy = nn.CrossEntropyLoss(weight=self.weights)

        # Attention Loss
        self.similarity_fn = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attention_loss = nn.MSELoss()

        # Consistency Regularization
        self.eps = 1e-6
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])

    def forward(self, pred, target, trans_feat):
        batch_size = pred.shape[0]

        # Weighted Cross Entropy Loss
        ce_loss = self.weighted_cross_entropy(pred, target)

        # Attention Loss
        attn_target = torch.zeros(batch_size, self.num_classes, device=self.device)
        attn_target[:, target] = 1
        attn_pred = F.softmax(pred, dim=1)
        attn_similarity = self.similarity_fn(trans_feat, attn_pred.detach())
        attn_loss = self.attention_loss(attn_similarity, attn_target)

        # Consistency Regularization
        perturbed_inputs = self.transform(trans_feat)
        perturbed_outputs = self.model(perturbed_inputs)
        perturbed_pred = F.softmax(perturbed_outputs, dim=1)
        consistency_similarity = self.similarity_fn(trans_feat, perturbed_pred.detach())
        consistency_loss = self.attention_loss(consistency_similarity, attn_pred.detach())

        total_loss = ce_loss + attn_loss + consistency_loss

        return total_loss
# --OPTION--