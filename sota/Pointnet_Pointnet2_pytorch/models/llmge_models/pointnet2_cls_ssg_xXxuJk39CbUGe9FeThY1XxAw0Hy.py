# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.optim.lr_scheduler as lr_scheduler

class GetModel(nn.Module):
    #... (rest of the code remains unchanged)

def train(model, optimizer, criterion, dataloader, device, epochs, batch_size, lr_decay_step, lr_gamma):
    model.train()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_gamma)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % batch_size == (batch_size - 1):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                     .format(epoch+1, epochs, i+1, len(dataloader), running_loss/batch_size))
                running_loss = 0.0

        scheduler.step()

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GetModel(num_class=10, normal_channel=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

train(model, optimizer, criterion, train_loader, device, epochs=100, batch_size=32, lr_decay_step=20, lr_gamma=0.1)
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--