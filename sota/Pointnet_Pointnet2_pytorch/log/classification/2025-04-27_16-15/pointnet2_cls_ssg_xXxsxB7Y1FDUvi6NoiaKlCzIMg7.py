# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import numpy as np

def calculate_average(numbers):
    if not numbers:
        return None
    return np.mean(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np
