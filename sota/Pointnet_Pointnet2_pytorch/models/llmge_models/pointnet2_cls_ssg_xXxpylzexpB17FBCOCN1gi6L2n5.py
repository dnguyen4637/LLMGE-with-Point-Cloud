# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def initialize_weights(*models):
    """Initialize weights for multiple models."""
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution layer."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def batch_norm(num_features):
    """Batch normalization layer."""
    return nn.BatchNorm1d(num_features)
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--