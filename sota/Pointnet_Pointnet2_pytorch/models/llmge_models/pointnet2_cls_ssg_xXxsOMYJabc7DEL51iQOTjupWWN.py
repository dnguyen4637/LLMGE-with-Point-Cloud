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
import random
import nltk
nltk.download('punkt')

class SimpleLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def tokenize(text):
    return nltk.word_tokenize(text)

def load_pretrained_embeddings(vocab_file, embedding_dim):
    vocab = {}
    embeddings = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector)!= embedding_dim:
                continue
            vocab[word] = len(embeddings)
            embeddings.append(vector)
    return vocab, np.vstack(embeddings)

def prepare_sequence(seq, word_idx_map, max_len):
    idxs = [word_idx_map[w] for w in seq]
    idxs = idxs[:max_len]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)

def prepare_data(texts, labels, word_idx_map, max_len):
    X = [prepare_sequence(txt, word_idx_map, max_len) for txt in texts]
    y = torch.tensor(labels, dtype=torch.long)
    return torch.stack(X), y

def get_random_batch(data, batch_size):
    rand_indices = np.random.choice(len(data[0]), size=batch_size, replace=False)
    X, y = data[0][rand_indices], data[1][rand_indices]
    return X, y

def train_epoch(model, iterator, optimizer, criterion, clip):
    epoch_loss = 0
    
    model.train()
    for i, batch in enumerate(
