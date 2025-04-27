
# ========== Start: GeneCrossed

# ========== End:
# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import torch.nn as nn
import torch.nn.functional as F

class GetLoss(nn.Module):
    def __init__(self):
        super(GetLoss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FeatureSimilarityLoss(nn.Module):
    def __init__(self, lambda_value=0.1):
        super().__init__()
        self.lambda_value = lambda_value

    def forward(self, pred_feats, true_feats):
        batch_size, channels, height, width = pred_feats.shape
        pred_feats = rearrange(pred_feats, 'b c h w -> b (h w) c')
        true_feats = rearrange(true_feats, 'b c h w -> b (h w) c')

        sim_matrix = torch.matmul(pred_feats, true_feats.T) / (height * width)
        sim_loss = -torch.logsumexp(sim_matrix, dim=-1) + torch.logsumexp(rearrange(pred_feats, 'b (h w) c -> b c (h w)'), dim=-1)

        return self.lambda_value * sim_loss.mean()

class KLDivLoss(nn.Module):
    def __init__(self, lambda_value=0.1):
        super().__init__()
        self.lambda_value = lambda_value

    def forward(self, mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= torch.numel(mu)

        return self.lambda_value * kld_loss

class GetLoss(nn.Module):
    def __init__(self):
        super(GetLoss, self).__init__()

    def forward(self, pred, target, trans_feat, pred_feats, true_feats):
        total_loss = F.nll_loss(pred, target)
        feature_sim_loss = FeatureSimilarityLoss()(pred_feats, true_feats)
        kl_div_loss = KLDivLoss()(trans_feat['mu'], trans_feat['logvar'])

        return total_loss + feature_sim_loss + kl_div_loss
# --OPTION--
import numpy as np

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
