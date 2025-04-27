# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.optim.lr_scheduler as lr_scheduler

class GetModel(nn.Module):
    #...

    def forward(self, xyz, norm=None):
        #...

        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1))
        x = self.fc3(x)
        x = F.log_softmax(x + 1e-5 * torch.randn_like(x), dim=-1)

        return x, l3_points

def train(model, optimizer, criterion, dataloader, device, epoch):
    model.train()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data, norm=None)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        #...
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--