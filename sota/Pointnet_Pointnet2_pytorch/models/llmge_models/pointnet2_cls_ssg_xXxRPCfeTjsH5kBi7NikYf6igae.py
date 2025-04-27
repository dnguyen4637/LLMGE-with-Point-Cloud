# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
self.bn1 = nn.BatchNorm1d(512)
self.bn2 = nn.BatchNorm1d(256)

#...

x = self.bn1(self.fc1(x))
x = F.relu(x)
x = self.drop1(x)

x = self.bn2(self.fc2(x))
x = F.relu(x)
x = self.drop2(x)
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--