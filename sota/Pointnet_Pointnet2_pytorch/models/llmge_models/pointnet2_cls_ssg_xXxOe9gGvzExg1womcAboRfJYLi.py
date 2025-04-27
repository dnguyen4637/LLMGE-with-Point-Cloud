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
