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
        self.read_weights = nn.ParameterList([nn.Parameter(torch.randn(num_heads, hidden_size)) for _ in range(num_heads)])
        self.write_weights = nn.ParameterList([nn.Parameter(torch.randn(num_heads, hidden_size)) for _ in range(num_heads)])
        self.shift_weights = nn.ParameterList([nn.Parameter(torch.randn(num_heads, hidden_size)) for _ in range(num_heads)])
        
    def forward(self, x):
        hc = self.controller(x, self.hc)
        c, h = hc
        
        read_vectors = torch.stack([F.linear(h, w) for w in self.read_weights], dim=0)
        read_weights = torch.softmax(read_vectors, dim=0)
        read_vector = torch.sum(read_weights * self.memory, dim=0)
        
        write_vectors = torch.stack([F.linear(c, w) for w in self.write_weights], dim=0)
        write_weights = torch.softmax(write_vectors, dim=0)
        erase_vector = torch.tanh(torch.stack([F.linear(c, w) for w in self.shift_weights], dim=0))
        
        self.memory = self.memory - erase_vector * self.memory + write_weights * torch.tanh(c)
        
        return read_vector, h

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, controller_hidden_size):
        super(Model, self).__init__()
        
        self.ntm = NeuralTuringMachine(input_size, hidden_size, num_heads, controller_hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        output, _ = self.ntm(x)
        output = self.fc(output)
        return output

model = Model(1, 10, 1, 2, 5)
input = torch.tensor([[1.0]])
output = model(input)
print(output)
