
# ========== Start: GeneCrossed

# ========== End:
# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F

class GetLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(GetLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, trans_feat=None):
        total_loss = F.nll_loss(pred, target)

        if trans_feat is not None:
            aux_loss = F.mse_loss(trans_feat, torch.zeros_like(trans_feat))
            total_loss += self.alpha * aux_loss

        return total_loss
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
