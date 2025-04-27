# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_cluster import fps

# Helper functions
def fps_helper(xyz, k=64):
    device = xyz.device
    idx = fps(xyz, k, batch_size=xyz.shape[0])
    idx = idx.to(device)
    return idx

def feature_propagation(features, idx, npoint):
    idx = idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    features = torch.gather(features, dim=0, index=idx)
    features = features.reshape(features.shape[0], npoint, -1)
    return features

def mlp(x, in_channels, hidden_channels, out_channels, dropout=0.4):
    x = nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_channels, out_channels),
        nn.ReLU()
    )(x)
    return x

class SpatialTransformerNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 64, kernel_size=1)
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, xyz):
        x = self.conv(xyz)
        x = x.permute(0, 2, 1)
        loc = self.fc_loc(x.mean(dim=-1))
        trans = torch.nn.functional.affine_grid(loc, xyz.size())
        xyz_transformed = torch.nn.functional.grid_sample(xyz, trans)
        return xyz_transformed

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_
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
        
    def forward(self, x):
        hc, c = self.controller(x, None)
        
        read_weights = self.read_weights(hc)
        read_weights = read_weights.view(-1, self.num_heads, self.hidden_size)
        read_weights = F.softmax(read_weights, dim=1)
        
        write_weights = self.write_weights(hc)
        write_weights = write_weights.view(-1, self.num_heads, self.hidden_size)
        shift_weights = self.shift_weights(hc)
        shift_weights = shift_weights.view(-1, self.num_heads, 2)
        
        erase_vectors = torch.sum(read_weights * self.memory, dim=1)
        add_vectors = torch.sum(write_weights * x.unsqueeze(1), dim=1)
        
        shifted_memory = torch.roll(self.memory, shifts=-shift_weights[:, :, 0].long(), dims=1)
        shifted_memory[:, -shift_weights[:, :, 1].long()] = 0
        
        self.memory = shifted_memory + erase_vectors * (1 - shift_weights[:, :, 1]) + add_vectors * shift_weights[:, :, 1]
        
        return self.memory

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.ntm = NeuralTuringMachine(hidden_size, hidden_size, num_heads=4, controller_hidden_size=hidden_size // 2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
