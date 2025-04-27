# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
from models.pointnetpp import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_class=40, normal_channel=True).to(device)

# Example input
xyz = torch.rand((2, 1024, 3)).to(device)

# Forward pass
logits, features = model(xyz)
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--