
# ========== Start: GeneCrossed

# ========== End:
# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        # Add any additional calculations or conditions here if needed

        return total_loss
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import torch
import torch.nn as nn
from torch.nn import functional as F

class NeuralTuringMachine(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, controller_hidden_size):
        super(NeuralTuringMachine, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.controller_hidden_size = controller_hidden_size
        
        self.controller = nn.LSTMCell(input_size, controller_hidden_size)
        self.memory = nn.Parameter(torch.randn(1, hidden_size))
        self.read_weights = nn.Linear(controller_hidden_size, num_heads * hidden_size)
        self.write_weights = nn.Linear(controller_hidden_size, num_heads * hidden_size)
        self.shift_weights = nn.Linear(controller_hidden_size, num_heads * 2)
        self.scale_factor = 0.1  # Scaling factor for erase/add vectors
        
    def forward(self, x):
        hc, c = self.controller(x, None)
        
        read_weights = self.read_weights(hc)
        read_weights = read_weights.view(-1, self.num_heads, self.hidden_size)
        read_weights = F.softmax(read_weights, dim=1)
        
        write_weights = self.write_weights(hc)
        write_weights = write_weights.view(-1, self.num_heads, self.hidden_size)
        shift_weights = self.shift_weights(hc)
        shift_weights = shift_weights.view(-1, self.num_heads, 2)
        
        erase_vectors = torch.sum(read_weights * self.memory, dim=1) * self.scale_factor
        add_vectors = torch
