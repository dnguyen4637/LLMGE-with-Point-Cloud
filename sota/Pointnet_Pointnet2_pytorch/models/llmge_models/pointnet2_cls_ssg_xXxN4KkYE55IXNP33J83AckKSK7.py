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
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms

# Define helper function for extracting features
def extract_features(model, image):
    # Set model to evaluation mode
    model.eval()

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(img_tensor)

    return features

# Define helper function for gram matrix calculation
def gram_matrix(feature_map):
    (b, ch, h, w) = feature_map.size()
    feature_map = feature_map.view(b, ch, h * w)
    feature_map = feature_map.transpose(1, 2)
    gram = torch.bmm(feature_map, feature_map.transpose(1, 2))
    return gram

# Define the new loss class
class ImprovedLoss(nn.Module):
    def __init__(self):
        super(ImprovedLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg.eval()

    def forward(self, pred, target, trans_feat):
        # Original NLL Loss
        total_loss = F.nll_loss(pred, target)

        # Feature Matching Loss
        vgg_target = extract_features(self.vgg, target)
        vgg_pred = extract_features(self.vgg, pred)
        feature_matching_loss = torch.mean(torch.abs(vgg_pred - vgg_target))

        # Style Transfer Loss
        gram_target = gram_matrix(vgg_target)
        gram_pred = gram_matrix(vgg_pred)
        style_transfer_loss = torch.mean((gram_pred - gram_target) ** 2)

        # Combine losses
        improved_loss = total_loss + 0.001 * feature_matching_loss + 0.0001 * style_transfer_loss

        return improved_loss
# --OPTION--