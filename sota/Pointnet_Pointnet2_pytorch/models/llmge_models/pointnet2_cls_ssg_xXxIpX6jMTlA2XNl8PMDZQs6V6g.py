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
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class PreTrainedEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        attn_weights = F.softmax(
            self.linear_out(torch.tanh(self.linear_in(hidden))), dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.embedding = PreTrainedEmbedding(embedding_dim, input_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        return outputs

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size
        self.embedding = PreTrainedEmbedding(embedding_dim, output_size)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        x = torch.cat((embedded, context), dim=-1)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(outputs[:, -1, :])
        return output, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
