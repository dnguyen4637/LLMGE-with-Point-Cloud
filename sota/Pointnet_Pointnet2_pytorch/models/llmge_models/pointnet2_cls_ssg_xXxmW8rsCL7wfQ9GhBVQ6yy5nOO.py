# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class GetModel(nn.Module):
    #... (same as before)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss / len(val_loader)}, Accuracy: {correct / total}")

# Example usage
model = GetModel(num_class=40, normal_channel=True)
criterion = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--