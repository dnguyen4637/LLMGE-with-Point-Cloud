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
import torchvision.transforms as transforms
from PIL import Image

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_feature = target_feature

    def forward(self, input_feature):
        gram_matrix = torch.mm(input_feature, input_feature.t()) / input_feature.size(1)
        loss = torch.mean((gram_matrix - self.target_feature) ** 2)
        return loss

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        loss = F.l1_loss(input, self.target)
        return loss

class GetLoss(nn.Module):
    def __init__(self, content_weight, style_weight):
        super(GetLoss, self).__init__()
        self.content_criterion = ContentLoss(target=None)
        self.style_criterion = StyleLoss(target_feature=None)
        self.content_weight = content_weight
        self.style_weight = style_weight

    def forward(self, pred, target, trans_feat):
        # Original NLL Loss
        total_loss = F.nll_loss(pred, target)

        # Feature Matching Loss
        content_loss = self.content_criterion(trans_feat, target) * self.content_weight

        # Style Transfer Loss
        style_loss = self.style_criterion(trans_feat, pred) * self.style_weight

        # Total Loss
        total_loss += content_loss + style_loss

        return total_loss

def load_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize(size, Image.ANTIALIAS)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    return image

# Example usage
content_img = load_image('path/to/content/image', (256, 256))
style_img = load_image('path/to/style/image', (256, 256))

model = YourModel()
criterion = GetLoss(content_weight=1, style_weight=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(content_img)
    loss = criterion(output, style_img, trans_feat=output)
    loss.backward()
    optimizer.step()
# --OPTION--